"""
Zigzag floor-coverage scan geometry and tuning. The layer that drives it
lives in src/subsumption/layers/layer1_scan.py.
"""
from .tuning import (
    FORWARD_SPEED, TURN_SPEED, TURN_AT_CM, TURN_90_S, SHIFT_S, MAX_LANE_S,
)

__all__ = [
    "FORWARD_SPEED", "TURN_SPEED", "TURN_AT_CM",
    "TURN_90_S", "SHIFT_S", "MAX_LANE_S",
]
