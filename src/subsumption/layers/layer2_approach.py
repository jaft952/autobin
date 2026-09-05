"""Layer 2: steers toward litter using the same error/command functions as tests/test_ibvs_centering.py."""
from typing import Any

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand
from src.visual_servoing.reactive_controller import compute_reactive_command

_CAL = MotionCalibration()
_KIN = DifferentialKinematics(_CAL)


def _to_motion_vector(wheels: WheelCommand):
    """Inverse of MotionExecutor's wheel mix, so re-mixing reproduces `wheels`."""
    scale = 2.0 * _CAL.forward_speed
    v_x = (wheels.left_speed + wheels.right_speed) / scale
    v_theta = (wheels.right_speed - wheels.left_speed) / scale
    return (v_x, 0, v_theta)


class ApproachLitterLayer(BaseLayer):
    """Layer 2, priority 2: arcs toward litter until Layer 3 takes over."""
    def __init__(self):
        super().__init__(layer_id=2)

    def evaluate(self, sensors: Any) -> ActionCommand:
        error = sensors.get_litter_target_error()
        if error is None or not error.found:
            return ActionCommand(layer_id=self.layer_id, active=False)

        wheels = compute_reactive_command(error, _KIN)
        if wheels is None:
            # reached: hold position so Layer 0 can't win before Layer 3 confirms grabbable
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
            arm_action='deploy',             # ready to grab
            message=label,
        )
