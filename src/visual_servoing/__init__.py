"""Visual servoing layer: perception -> target error -> drive command.

Two controllers are available:
    compute_reactive_command       -- original reactive (IBVS) controller
    compute_predictive_command  -- receding-horizon predictive planner
                                    (prediction_model / trajectory_planner /
                                    cost_function / predictive_controller)
"""

from src.visual_servoing.distance_error import TargetError, compute_target_error
from src.visual_servoing.reactive_controller import compute_reactive_command
from src.visual_servoing.predictive.prediction_model import ActionStep, RobotState, TargetPoint
from src.visual_servoing.predictive.trajectory_planner import Plan
from src.visual_servoing.predictive_controller import ControllerState, compute_predictive_command

__all__ = [
    "TargetError",
    "compute_target_error",
    "compute_reactive_command",
    "ActionStep",
    "RobotState",
    "TargetPoint",
    "Plan",
    "ControllerState",
    "compute_predictive_command",
]
