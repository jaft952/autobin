"""
src/subsumption/layers/layer2_approach.py

Layer 2: Approach Litter -- owns every base movement made while closing on a
tin, and decides when the base has arrived.

Steering is the SAME two functions, on the SAME data, as tests/test_ibvs_
centering.py's validated live loop:

    error  = sensors.get_litter_target_error()   # compute_target_error()
    wheels = compute_reactive_command(error, kin) # arc steering + speed tiers

ARRIVAL is judged by the arc grid (src/arm/grasp_reach.py), NOT by the
monocular distance estimate. Those are different measurements: the base used
to park on distance_cm <= STOP_DISTANCE_CM while Layer 3 read the same spot
off the grid and called it an overshoot, so the two layers drove the base
back and forth against each other. One criterion, owned here, ends that --
and the overshoot back-off lives here too, because backing up is base
movement and Layer 3 has no business steering.

By the time Layer 3 wins arbitration the base is already stopped and braked,
so the grab does not start on a chassis that is still rolling.

compute_reactive_command() returns a WheelCommand already in duty units (the
~20-32 scale PWMActuator expects directly), not the -1..1 motion fraction
this layer must emit (Rule 1: hardware mixing happens once, in
MotionExecutor, after arbitration -- a layer never touches wheel-level
values). _to_motion_vector() undoes MotionExecutor's own mixing formula, so
when MotionExecutor re-mixes the vector this layer emits, it reproduces the
exact duty compute_reactive_command() intended.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.arm.arc_grasp import ArcGraspSolver, BAND_TOO_CLOSE
from src.arm.grasp_reach import solve_reach
from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand
from src.visual_servoing.reactive_controller import compute_reactive_command, BACKUP_SPEED

_CAL = MotionCalibration()
_KIN = DifferentialKinematics(_CAL)

RETREAT_GAP_S = 0.05
RETREAT_PULSE_S = 0.3
# A one-frame detection blink must not abandon a retreat half-way: Layer 2
# would go inactive, Layer 0 would win, and the base would COAST away from
# the spot mid-manoeuvre.
LOST_GRACE_S = 1.0


def _to_motion_vector(wheels: WheelCommand):
    """Inverse of MotionExecutor's left = v_x - v_theta, right = v_x +
    v_theta mix, so re-mixing this vector reproduces `wheels` exactly
    (peak stays under the renormalize threshold at these duty levels)."""
    scale = 2.0 * _CAL.forward_speed
    v_x = (wheels.left_speed + wheels.right_speed) / scale
    v_theta = (wheels.right_speed - wheels.left_speed) / scale
    return (v_x, 0, v_theta)


class ApproachLitterLayer(BaseLayer):
    """
    Layer 2: Approach Litter
    Priority: 2 (Low)
    Behavior: Arcs the base toward detected ground litter, backs off an
              overshoot, and holds the spot once the arc grid can solve the
              tin from where the base stands. Layer 3 then grabs without
              moving the base at all.
    """

    def __init__(self, arc_solver: Optional[ArcGraspSolver] = None):
        super().__init__(layer_id=2)
        self.solver = arc_solver or ArcGraspSolver()
        self._retreating: bool = False   # True = mid-pulse, False = mid-gap
        self._retreat_until: float = 0.0
        self._was_too_close: bool = False
        self._lost_until: float = 0.0    # blink grace while backing off
        self._suppressed_since: Optional[float] = None

    def reset(self) -> None:
        self._retreating = False
        self._retreat_until = 0.0
        self._was_too_close = False
        self._lost_until = 0.0
        self._suppressed_since = None

    # ── Arbitration ───────────────────────────────────────────────────────

    def notify_arbitration(self, won: bool) -> None:
        if not won and self._suppressed_since is None:
            self._suppressed_since = time.monotonic()

    def evaluate(self, sensors: Any) -> ActionCommand:
        now = time.monotonic()
        self._absorb_suppressed_time(now)

        error = sensors.get_litter_target_error()
        if error is None or not error.found:
            if self._was_too_close and now < self._lost_until:
                # Mid-retreat blink. Stop the pulse -- reversing blind is not
                # worth it -- but keep the manoeuvre alive so the next good
                # frame resumes it instead of restarting from the gap.
                return self._coast("OVERSHOT - waiting for the detection")
            self._was_too_close = False
            return ActionCommand(layer_id=self.layer_id, active=False)
        self._lost_until = now + LOST_GRACE_S

        reach = solve_reach(self.solver, sensors)

        if reach.band == BAND_TOO_CLOSE:
            return self._retreat_pulse(now)
        self._was_too_close = False

        if reach.pose is not None:
            # Arrived. Brake and stay put: Layer 3 wins from here, and it
            # should inherit a base that has already stopped.
            nx, ny = reach.point # type: ignore
            return self._hold(f"IN REACH, holding ({reach.klass} "
                              f"nx={nx:.2f} ny={ny:.2f})")

        wheels = compute_reactive_command(error, _KIN)
        if wheels is None:
            # Vision says centered and close, but the grid cannot solve this
            # spot (uncalibrated, or the tin sits off the sampled span).
            # Hold rather than keep driving on a criterion the arm ignores.
            return self._hold(f"REACHED but not in reach "
                              f"(dist={error.distance_cm}, band={reach.band})")

        label = "TOO CLOSE, backing off" if error.too_close else \
            f"APPROACHING LITTER (ex={error.lateral_error:+.2f}, dist={error.distance_cm})"
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=_to_motion_vector(wheels),
            arm_action='deploy',             # travel pose, ready to grab
            message=label,
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _absorb_suppressed_time(self, now: float) -> None:
        """Open-loop timers must not count time spent suppressed by a higher
        layer -- the retreat pulse would run its phases while the base was
        being driven by somebody else."""
        if self._suppressed_since is None:
            return
        paused = now - self._suppressed_since
        if self._retreat_until:
            self._retreat_until += paused
        if self._lost_until:
            self._lost_until += paused
        self._suppressed_since = None

    def _retreat_pulse(self, now: float) -> ActionCommand:
        """Bounded backward pulse with a brief stopped gap before it, same
        shape as test_ibvs_centering.py's _retreat_pulse() (stop -> backward
        at BACKUP_SPEED for RETREAT_PULSE_S -> stop), re-expressed as a timed
        phase instead of blocking sleeps."""
        if not self._was_too_close:
            # Freshly entered BAND_TOO_CLOSE: always start with the stopped
            # gap, never straight into reverse (hardware_safety_patterns.md
            # rule 7 -- a direction flip needs a stop in between).
            self._was_too_close = True
            self._retreating = False
            self._retreat_until = now + RETREAT_GAP_S
        elif now >= self._retreat_until:
            self._retreating = not self._retreating
            self._retreat_until = now + (RETREAT_PULSE_S if self._retreating
                                         else RETREAT_GAP_S)

        if self._retreating:
            v_x = -BACKUP_SPEED / _CAL.forward_speed
            return ActionCommand(
                layer_id=self.layer_id, active=True,
                motion_vector=(v_x, 0, 0), arm_action='deploy',
                message="OVERSHOT past nearest arc - retreat pulse",
            )
        # COAST, not brake. This gap exists so a direction flip never goes
        # straight from forward to reverse; braking here would drive both
        # inputs of each motor HIGH, which is the very pulse the gap is meant
        # to avoid, once every RETREAT_PULSE_S.
        return self._coast("OVERSHOT past nearest arc - pulse gap")

    def _coast(self, message: str) -> ActionCommand:
        """Cut power and let the base roll to rest. 'deploy' is not in
        MotionExecutor._GRAB_ACTIONS, so a zero vector under it coasts."""
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),
            arm_action='deploy',
            message=message,
        )

    def _hold(self, message: str) -> ActionCommand:
        """Stay on this spot. 'hold' brakes the wheels (see
        MotionExecutor._GRAB_ACTIONS); 'deploy' would coast, and a coasting
        base rolls off the spot the grid just solved for."""
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),
            arm_action='hold',
            message=message,
        )
