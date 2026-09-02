"""
Zigzag floor-coverage scan: no odometry, so the base reacts to the
ultrasonic instead of following a map. See layer.py for the full picture.
"""
from .layer import ScanAroundLayer
from .tuning import (
    FORWARD_SPEED, TURN_SPEED, TURN_AT_CM, TURN_90_S, SHIFT_S, MAX_LANE_S,
)

__all__ = [
    "ScanAroundLayer",
    "FORWARD_SPEED", "TURN_SPEED", "TURN_AT_CM",
    "TURN_90_S", "SHIFT_S", "MAX_LANE_S",
]
