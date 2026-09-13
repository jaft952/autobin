"""Turns the steering error into wheel speeds."""
from __future__ import annotations

from typing import Optional

from src.visual_servoing.distance_error import TargetError
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand

FAR_DISTANCE_CM = 100.0
LOW_DISTANCE_CM = 50.0

FORWARD_HIGH_SPEED = 32.0
FORWARD_MID_SPEED = 30.0
FORWARD_LOW_SPEED = 28.0

BACKUP_SPEED = 20.0
MAX_STEER_ANGLE_DEG = 90.0


def compute_reactive_command(error: TargetError,
                           kin: DifferentialKinematics) -> Optional[WheelCommand]:
    """Wheel command from the steering error, using fixed speed steps."""
    if not error.found:
        return None

    if error.too_close:
        return kin.backward(speed=BACKUP_SPEED)

    if error.reached:
        return None

    steer_angle = min(MAX_STEER_ANGLE_DEG,
                       abs(error.lateral_error) * 2 * MAX_STEER_ANGLE_DEG)

    if error.distance_cm is None:
        dynamic_speed = 0
    elif error.distance_cm > FAR_DISTANCE_CM:
        dynamic_speed = FORWARD_HIGH_SPEED
    elif error.distance_cm > LOW_DISTANCE_CM and error.distance_cm <= FAR_DISTANCE_CM:
        dynamic_speed = FORWARD_MID_SPEED
    elif error.distance_cm > 0 and error.distance_cm <= LOW_DISTANCE_CM:
        dynamic_speed = FORWARD_LOW_SPEED
    else:
        dynamic_speed = 0

    if error.lateral_error > 0:
        return kin.arc_forward_right(angle_deg=steer_angle, speed=dynamic_speed)
    else:
        return kin.arc_forward_left(angle_deg=steer_angle, speed=dynamic_speed)


def speed_tier(error: TargetError) -> str:
    """Name of the speed step this error falls into."""
    if not error.found:
        return "none"
    if error.too_close:
        return "backup"
    if error.reached:
        return "reached"
    if error.distance_cm is None:
        return "fallback"
    if error.distance_cm > FAR_DISTANCE_CM:
        return "cruise"
    if error.distance_cm > LOW_DISTANCE_CM and error.distance_cm <= FAR_DISTANCE_CM:
        return "approach"
    if error.distance_cm > 0 and error.distance_cm <= LOW_DISTANCE_CM:
        return "step"
    return "fallback"