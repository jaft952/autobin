"""
src/subsumption/layers/layer2_approach.py

Layer 2: Approach Litter -- steers the base toward detected ground litter
by calling the SAME two functions, on the SAME data, as tests/test_ibvs_
centering.py's validated live loop:

    error  = sensors.get_litter_target_error()   # compute_target_error()
    wheels = compute_reactive_command(error, kin) # arc steering + speed tiers

get_litter_target_error() (CameraSensor) calls compute_target_error() on
its own locked-target DetectionResult -- it is not a value re-derived from
this layer's other getters, so there is nothing here that can drift out of
sync with the validated script: same function, same input, same constants.

compute_reactive_command() returns a WheelCommand already in duty units
(the ~20-32 scale PWMActuator expects directly), not the -1..1 motion
fraction this layer must emit (Rule 1: hardware mixing happens once, in
MotionExecutor, after arbitration -- a layer never touches wheel-level
values). _to_motion_vector() undoes MotionExecutor's own mixing formula, so
when MotionExecutor re-mixes the vector this layer emits, it reproduces the
exact duty compute_reactive_command() intended.
"""
from typing import Any

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand
from src.visual_servoing.reactive_controller import compute_reactive_command

_CAL = MotionCalibration()
_KIN = DifferentialKinematics(_CAL)


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
    Behavior: Arcs the base toward detected ground litter using the same
              TargetError -> compute_reactive_command() pipeline validated
              in tests/test_ibvs_centering.py, until Layer 3 finds the tin
              grabbable and takes over.
    """
    def __init__(self):
        super().__init__(layer_id=2)

    def evaluate(self, sensors: Any) -> ActionCommand:
        error = sensors.get_litter_target_error()
        if error is None or not error.found:
            return ActionCommand(layer_id=self.layer_id, active=False)

        wheels = compute_reactive_command(error, _KIN)
        if wheels is None:
            # Reached: within STOP_DISTANCE_CM and centered. Hold position
            # (stay active, do not go inactive) so Layer 0 Idle cannot win
            # the gap before Layer 3 independently confirms grabbable.
            return ActionCommand(
                layer_id=self.layer_id,
                active=True,
                motion_vector=(0, 0, 0),
                arm_action='deploy',
                message=f"REACHED, holding (dist={error.distance_cm})",
            )

        label = "TOO CLOSE, backing off" if error.too_close else \
            f"APPROACHING LITTER (ex={error.lateral_error:+.2f}, dist={error.distance_cm})"
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=_to_motion_vector(wheels),
            arm_action='deploy',             # travel pose, ready to grab
            message=label,
        )
