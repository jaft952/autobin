"""
src/visual_servoing/predictive_controller.py

Predictive (receding-horizon) replacement for approach_drive.compute_reactive_command.
Every call: run trajectory_planner.plan() to search HORIZON_STEPS ahead, then
execute ONLY the first action of the best sequence and throw the rest of the
plan away -- the next call re-plans from scratch against that frame's fresh
TargetError. See trajectory_planner.py for the search and cost_function.py
for the scoring; see prediction_model.py for the physics and the explanation
of why the frame resets every cycle (no odometry/IMU on this robot).

The not-found / too-close / reached short-circuits are copied byte-for-byte
from approach_drive.py: those are safety failsafes, not planning decisions,
so they bypass the search entirely, same as the reactive controller.

Pure-function style like the rest of visual_servoing/: no hidden mutable
controller object. The one piece of state this needs across calls -- the
previous action, for cost_function.py's oscillation terms -- is threaded
explicitly through ControllerState rather than stashed on self.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.visual_servoing.distance_error import TargetError
from visual_servoing.reactive_controller import BACKUP_SPEED
from visual_servoing.predictive.prediction_model import ActionStep
from visual_servoing.predictive.trajectory_planner import plan, action_to_command
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand


@dataclass
class ControllerState:
    prev_action: ActionStep = field(default_factory=lambda: ActionStep(steer_deg=0.0, speed=0.0))


def compute_predictive_command(
    error: TargetError, kin: DifferentialKinematics, state: ControllerState,
) -> Tuple[Optional[WheelCommand], ControllerState]:
    """Drop-in predictive counterpart to approach_drive.compute_reactive_command,
    with one API difference: it needs the CALLER's previous ControllerState
    back each cycle (a bare TargetError isn't enough once oscillation is part
    of the cost). Callers hold one ControllerState for the lifetime of the
    approach, e.g.:

        state = ControllerState()
        while True:
            error = compute_target_error(detector.detect())
            cmd, state = compute_predictive_command(error, kin, state)
            actuator.apply(cmd) if cmd is not None else actuator.stop()

    Short-circuit branches leave `state` untouched -- they didn't plan this
    cycle, so there's nothing new to remember as "the previous action".
    """
    if not error.found:
        return None, state

    if error.too_close:
        return kin.backward(speed=BACKUP_SPEED), state

    if error.reached:
        return None, state

    result = plan(error, kin, state.prev_action)
    first_action = result.actions[0]
    cmd = action_to_command(first_action, kin)
    return cmd, ControllerState(prev_action=first_action)
