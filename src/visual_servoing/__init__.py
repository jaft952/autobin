"""Visual servoing: turn the can's image position into wheel commands."""

from src.visual_servoing.distance_error import TargetError, compute_target_error
from src.visual_servoing.reactive_controller import (
    compute_reactive_command, speed_tier,
)

__all__ = [
    "TargetError",
    "compute_target_error",
    "compute_reactive_command",
    "speed_tier",
]
