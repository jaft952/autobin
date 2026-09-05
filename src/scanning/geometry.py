"""Pure sensor-reading helpers for the scan layer. No state, no time."""
import random
from typing import Any

from . import tuning


def jitter(seconds: float) -> float:
    if tuning.TIMING_JITTER <= 0.0:
        return seconds
    return seconds * random.uniform(1.0 - tuning.TIMING_JITTER, 1.0 + tuning.TIMING_JITTER)


def fmt(value) -> str:
    return "--" if value is None else f"{value:.0f}"


def side_room_cm(sensors: Any, getter: str) -> float:
    """Free space on one side, capped at PIVOT_ROOM_CM. None means open."""
    fn = getattr(sensors, getter, None)
    value = fn() if fn is not None else None
    return tuning.PIVOT_ROOM_CM if value is None else min(value, tuning.PIVOT_ROOM_CM)


def clearances(sensors: Any):
    """(front, front_left, front_right) in cm; None = nothing in range."""
    out = []
    for name in ("get_obstacle_distance_cm",
                 "get_obstacle_distance_front_left_cm",
                 "get_obstacle_distance_front_right_cm"):
        getter = getattr(sensors, name, None)
        out.append(getter() if getter is not None else None)
    return tuple(out)


def lane_bias(left, right, turn_speed: float) -> float:
    """Steer away from a close side wall; opposing walls cancel out."""
    bias = 0.0
    if left is not None and left < tuning.DIAGONAL_NUDGE_CM:
        bias -= tuning.NUDGE_GAIN * (1.0 - left / tuning.DIAGONAL_NUDGE_CM)
    if right is not None and right < tuning.DIAGONAL_NUDGE_CM:
        bias += tuning.NUDGE_GAIN * (1.0 - right / tuning.DIAGONAL_NUDGE_CM)
    return max(-1.0, min(1.0, bias)) * turn_speed


def pivot_side(sensors: Any, turn_left: bool):
    """Pick pivot side. Keep alternating unless that side is tight. Returns (turn_left, note)."""
    left = side_room_cm(sensors, "get_obstacle_distance_front_left_cm")
    right = side_room_cm(sensors, "get_obstacle_distance_front_right_cm")
    note = f"L{left:.0f} R{right:.0f}"
    intended, other = (left, right) if turn_left else (right, left)
    if intended < tuning.PIVOT_TIGHT_CM and other >= intended + tuning.PIVOT_MARGIN_CM:
        return not turn_left, note
    return turn_left, note
