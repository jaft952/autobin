"""
src/visual_servoing/predictive/trajectory_planner.py

Predictive planning without search: instead of scoring candidate action
sequences against a hand-tuned cost function, this rolls ONE smooth
speed/steer profile forward through prediction_model.step() until the
target's predicted position is reached. There is only one candidate at each
step -- the profile itself -- so there is nothing to search over and no
weights to tune against each other.

The profile is shaped by distance-to-go only, and is rate-limited so it
never jumps:

    far away        -> ramps UP toward cruise speed (MAX_SPEED), capped by
                        MAX_ACCEL_PER_S so the first command after a
                        standing start is gentle, not a speed jump
    mid-approach     -> holds cruise speed
    near the target  -> eases DOWN (smoothstep, not linear) as distance_cm
                        falls from FAR_DISTANCE_CM to STOP_DISTANCE_CM,
                        capped by MAX_DECEL_PER_S, arriving at ~zero speed
                        exactly at the grasp distance
Steering follows the same rate-limited-ramp idea off lateral_error, so the
arc angle doesn't jump step to step either -- that's what replaces the old
cost function's oscillation penalty.

FAR_DISTANCE_CM / STOP_DISTANCE_CM / MAX_SPEED / MAX_STEER_ANGLE_DEG are
reused from reactive_controller.py / distance_error.py rather than redefined
here, so the reactive and predictive controllers agree on "far", "close
enough to grasp" and the speed/steer ceilings -- one tuned set of numbers,
not two that could drift apart.

Every planned step is converted to a WheelCommand through the SAME
DifferentialKinematics.arc_forward_left/right methods the real hardware is
driven with (action_to_command, below), so whatever this plans is guaranteed
physically reachable -- there is no separate "planning model" that could
drift from what actually gets executed.

Like the beam-search version this replaces, the full rollout (a Journey) is
mostly a PREVIEW: predictive_controller.py only ever executes steps[0] and
re-plans from scratch next camera frame (see that module's docstring for why
-- no odometry/IMU on this robot). The rest of the journey is still useful to
look at -- ETA to the grasp point, whether the profile is behaving smoothly
-- which is why plan_journey() returns the whole thing instead of just one
step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from src.visual_servoing.distance_error import TargetError, STOP_DISTANCE_CM, CENTER_TOLERANCE
from ..reactive_controller import (
    MAX_SPEED, MAX_STEER_ANGLE_DEG,
    CRUISE_DISTANCE_CM as FAR_DISTANCE_CM,   # renamed in the reactive-only refactor
)
from .prediction_model import (
    ActionStep, CONTROL_DT_S,
    estimate_initial_state, step as physics_step, predicted_error,
)
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand

# ── Smooth speed profile ─────────────────────────────────────────────────
MAX_ACCEL_PER_S = 60.0        # TODO tune: speed-units/s the ramp may climb (the "slow start" feel)
MAX_DECEL_PER_S = 90.0        # TODO tune: speed-units/s the ramp may fall (the "ease to a stop" feel)

# Creep-speed floor for the final approach (STOP_DISTANCE_CM < distance <
# FAR_DISTANCE_CM), so the profile keeps making real forward progress
# instead of asymptotically decaying toward STOP_DISTANCE_CM without ever
# crossing it (speed -> 0 as distance -> STOP_DISTANCE_CM means, taken
# literally, the last few cm take forever to close). Same "motors stall
# below this" reasoning as layer2_approach.APPROACH_MIN_SPEED elsewhere in
# this codebase. Speed only reaches true 0 once distance_cm has actually
# crossed STOP_DISTANCE_CM (distance_error.reached takes over from there).
MIN_APPROACH_SPEED = 4.0      # TODO tune

# ── Smooth steer profile ─────────────────────────────────────────────────
MAX_STEER_RATE_DEG_S = 120.0  # TODO tune: deg/s the arc angle is allowed to change

# Safety cap on how far into the future one journey is rolled out. At
# CONTROL_DT_S=0.2s this is ~15s of predicted travel -- generous for any
# realistic approach distance. It only guards against a journey that can't
# converge (e.g. a lateral error beyond MAX_STEER_ANGLE_DEG's authority)
# rolling forward forever; a converging journey stops far short of it.
MAX_JOURNEY_STEPS = 75        # TODO tune


@dataclass
class JourneyStep:
    """One planned motion command: how far to arc (degrees, same sign
    convention as TargetError.lateral_error), how fast, and for how long --
    the three numbers action_to_command/DifferentialKinematics need."""
    arc_deg: float
    speed: float
    duration_s: float
    predicted_lateral_error: float
    predicted_distance_cm: float


@dataclass
class Journey:
    """The full planned approach, from the current camera frame to the
    grasp point. Only steps[0] is ever actually executed -- see the module
    docstring -- the rest is a preview for logging/inspection."""
    steps: List[JourneyStep] = field(default_factory=list)
    arrived: bool = False   # rollout reached the grasp point within MAX_JOURNEY_STEPS

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.steps)


def action_to_command(action: ActionStep, kin: DifferentialKinematics) -> WheelCommand:
    """Hardware-verified mapping copied from reactive_controller.compute_reactive_command:
    +ve steer (correcting toward a can right of center) -> arc_forward_LEFT,
    -ve -> arc_forward_RIGHT. Do not re-derive this from geometry -- it was
    flipped from the naive reading once already on real hardware.

    Exactly 0 -> kin.forward(), NOT arc_forward_left(angle_deg=0.0). The arc
    methods bake an asymmetric per-wheel trim directly into the command
    (apply_trim=False) that's tuned for arcing, not for a literal straight
    line -- at angle_deg=0 that trim alone puts left/right ~13% apart on this
    robot's calibration, which reads as a small constant turn. Over a single
    reactive tick that's negligible, but plan_journey() below chains dozens
    of ticks together, so that bias compounds into real drift. forward()
    commands equal left/right and lets the actuator's own apply_trim path
    correct it instead, which is what "no steering" actually means here."""
    if action.steer_deg > 0:
        return kin.arc_forward_left(angle_deg=abs(action.steer_deg), speed=action.speed)
    if action.steer_deg < 0:
        return kin.arc_forward_right(angle_deg=abs(action.steer_deg), speed=action.speed)
    return kin.forward(speed=action.speed)


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _target_speed(distance_cm: float, lateral_error: float) -> float:
    """Desired speed for this (distance, lateral_error) pair -- NOT rate-
    limited yet, see _ramp(). Smoothstep (not linear) between
    FAR_DISTANCE_CM and STOP_DISTANCE_CM so the deceleration itself feels
    gradual instead of a straight-line ramp.

    Only drops all the way to 0 once truly arrived (close AND centered).
    Speed and steer both flow through the SAME WheelCommand, so a literal 0
    here would zero out the arc's wheel differential too -- if the target
    is still off-center exactly when distance_cm first crosses
    STOP_DISTANCE_CM, that would freeze the robot facing the wrong way
    just short of the grasp point, with nothing left to turn it the rest
    of the way (this is a known limitation documented on
    reactive_controller.compute_reactive_command, which shares the same
    coupling). Holding MIN_APPROACH_SPEED instead keeps just enough wheel
    differential alive to finish centering before the distance ramp would
    otherwise starve it.
    """
    if distance_cm <= STOP_DISTANCE_CM and abs(lateral_error) <= CENTER_TOLERANCE:
        return 0.0
    if distance_cm >= FAR_DISTANCE_CM:
        return MAX_SPEED
    if distance_cm <= STOP_DISTANCE_CM:
        return MIN_APPROACH_SPEED
    span = FAR_DISTANCE_CM - STOP_DISTANCE_CM
    frac = (distance_cm - STOP_DISTANCE_CM) / span
    return max(MIN_APPROACH_SPEED, MAX_SPEED * _smoothstep(frac))


def _target_steer_for_lateral_error(lateral_error: float) -> float:
    """Same proportional law as reactive_controller.compute_reactive_command,
    NOT rate-limited yet -- see _ramp()."""
    if lateral_error == 0:
        return 0.0
    magnitude = min(MAX_STEER_ANGLE_DEG, abs(lateral_error) * 2 * MAX_STEER_ANGLE_DEG)
    return math.copysign(magnitude, lateral_error)


def _ramp(current: float, target: float, max_delta: float) -> float:
    """Move `current` toward `target`, capped to +-max_delta per step -- the
    one mechanism behind both the speed and steer profiles' smoothness. This
    is what turns a step target (e.g. speed jumping straight to MAX_SPEED the
    instant the target is far enough away) into a gradual ramp."""
    delta = max(-max_delta, min(max_delta, target - current))
    return current + delta


def plan_journey(error: TargetError, kin: DifferentialKinematics, prev_action: ActionStep,
                  dt_s: float = CONTROL_DT_S, max_steps: int = MAX_JOURNEY_STEPS) -> Journey:
    """Roll the smooth speed/steer profile forward, one CONTROL_DT_S tick at
    a time, from the robot's current (local-frame) position until the
    target's predicted position is reached (predicted_distance_cm <=
    STOP_DISTANCE_CM and centered) or max_steps is hit.

    Precondition (caller's responsibility, same as reactive_controller.py's
    use of TargetError): error.found is True, error.too_close and
    error.reached are both False -- predictive_controller.py short-circuits
    those cases before this ever runs.
    """
    state, target = estimate_initial_state(error)
    speed, steer = prev_action.speed, prev_action.steer_deg

    steps: List[JourneyStep] = []
    for _ in range(max_steps):
        lateral_error, distance_cm = predicted_error(state, target)
        if distance_cm <= STOP_DISTANCE_CM and abs(lateral_error) <= CENTER_TOLERANCE:
            return Journey(steps=steps, arrived=True)

        target_steer = _target_steer_for_lateral_error(lateral_error)
        target_speed = _target_speed(distance_cm, lateral_error)
        accel_limit = MAX_ACCEL_PER_S if target_speed >= speed else MAX_DECEL_PER_S
        speed = _ramp(speed, target_speed, accel_limit * dt_s)
        steer = _ramp(steer, target_steer, MAX_STEER_RATE_DEG_S * dt_s)

        action = ActionStep(steer_deg=steer, speed=speed)
        state = physics_step(state, action_to_command(action, kin), dt_s, kin.cal)

        steps.append(JourneyStep(arc_deg=steer, speed=speed, duration_s=dt_s,
                                  predicted_lateral_error=lateral_error,
                                  predicted_distance_cm=distance_cm))

    return Journey(steps=steps, arrived=False)
