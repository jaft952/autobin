"""Visual servoing layer: perception -> target error -> drive command.

Two controllers are available:
    compute_drive_command       -- original reactive (IBVS) controller
    compute_predictive_command  -- receding-horizon predictive planner
                                    (prediction_model / trajectory_planner /
                                    cost_function / predictive_controller)
"""

from src.visual_servoing.distance_error import TargetError, compute_target_error
from src.visual_servoing.approach_drive import compute_drive_command
from src.visual_servoing.prediction_model import ActionStep, RobotState, TargetPoint
from src.visual_servoing.trajectory_planner import Plan
from src.visual_servoing.predictive_controller import ControllerState, compute_predictive_command

__all__ = [
    "TargetError",
    "compute_target_error",
    "compute_drive_command",
    "ActionStep",
    "RobotState",
    "TargetPoint",
    "Plan",
    "ControllerState",
    "compute_predictive_command",
]
