"""Visual servoing layer: perception -> target error -> drive command.

Two controllers are available:
    compute_reactive_command     -- original reactive (IBVS) controller
    compute_predictive_command   -- receding-horizon predictive planner
                                    (predictive/prediction_model.py +
                                    predictive/trajectory_planner.py +
                                    predictive_controller.py)
"""

from src.visual_servoing.distance_error import TargetError, compute_target_error
from src.visual_servoing.reactive_controller import (
    compute_reactive_command, is_final_approach, speed_tier,
)
from src.visual_servoing.predictive_controller import (
    compute_predictive_command, ControllerState,
)

__all__ = [
    "TargetError",
    "compute_target_error",
    "compute_reactive_command",
    "is_final_approach",
    "speed_tier",
    "compute_predictive_command",
    "ControllerState",
]
