"""
src/visual_servoing/predictive_controller.py

Predictive (receding-horizon) replacement for reactive_controller.compute_reactive_command.
Every call: run trajectory_planner.plan_journey() to roll a smooth speed/
steer profile forward to the grasp point (see that module for the profile
itself -- slow start, cruise, gradual stop), then execute ONLY the first
step of that journey and throw the rest away -- the next call re-plans from
scratch against that frame's fresh TargetError. See predictive/
prediction_model.py for the physics and the explanation of why the frame
resets every cycle (no odometry/IMU on this robot).

The not-found / too-close / reached short-circuits are copied byte-for-byte
from reactive_controller.py: those are safety failsafes, not planning decisions,
so they bypass planning entirely, same as the reactive controller.

Pure-function style like the rest of visual_servoing/: no hidden mutable
controller object. The state this needs across calls -- the previous action
(so the speed/steer ramp in trajectory_planner.py doesn't jump step to
step), plus the last full journey (so a caller can log/display the plan,
e.g. an ETA to the grasp point) -- is threaded explicitly through
ControllerState rather than stashed on self.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.visual_servoing.distance_error import TargetError
from .reactive_controller import BACKUP_SPEED
from .predictive.prediction_model import ActionStep
from .predictive.trajectory_planner import plan_journey, action_to_command, Journey
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand


@dataclass
class ControllerState:
    prev_action: ActionStep = field(default_factory=lambda: ActionStep(steer_deg=0.0, speed=0.0))
    last_journey: Optional[Journey] = None


def compute_predictive_command(
    error: TargetError, kin: DifferentialKinematics, state: ControllerState,
) -> Tuple[Optional[WheelCommand], ControllerState]:
    """Drop-in predictive counterpart to reactive_controller.compute_reactive_command,
    with one API difference: it needs the CALLER's previous ControllerState
    back each cycle (a bare TargetError isn't enough once the speed/steer
    ramp depends on what was commanded last cycle). Callers hold one
    ControllerState for the lifetime of the approach, e.g.:

        state = ControllerState()
        while True:
            error = compute_target_error(detector.detect())
            cmd, state = compute_predictive_command(error, kin, state)
            actuator.apply(cmd) if cmd is not None else actuator.stop()
            # state.last_journey.steps -> full planned approach, for logging

    Short-circuit branches leave `state` untouched -- they didn't plan this
    cycle, so there's nothing new to remember as "the previous action".
    """
    if not error.found:
        return None, state

    if error.too_close:
        return kin.backward(speed=BACKUP_SPEED), state

    if error.reached:
        return None, state

    journey = plan_journey(error, kin, state.prev_action)
    if not journey.steps:
        return None, state

    first = journey.steps[0]
    first_action = ActionStep(steer_deg=first.arc_deg, speed=first.speed)
    cmd = action_to_command(first_action, kin)
    return cmd, ControllerState(prev_action=first_action, last_journey=journey)
