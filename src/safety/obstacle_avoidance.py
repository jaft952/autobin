"""Pure obstacle-avoidance decisions: (readings, thresholds) -> a decision, no layer/state knowledge."""
from __future__ import annotations

from typing import Optional


def is_wall_ahead(front_cm: Optional[float], threshold_cm: float) -> bool:
    return front_cm is not None and front_cm <= threshold_cm


def is_clear(distance_cm: float, threshold_cm: float) -> bool:
    return distance_cm >= threshold_cm


def room_favors_left(left_cm: float, right_cm: float) -> bool:
    return left_cm >= right_cm


def pick_pivot_side(turning_left: bool, intended_room_cm: float,
                     other_room_cm: float, *, tight_cm: float,
                     margin_cm: float) -> bool:
    """Only break alternation if intended side is tight and the other is clearly roomier."""
    if intended_room_cm < tight_cm and other_room_cm >= intended_room_cm + margin_cm:
        return not turning_left
    return turning_left


def lane_bias(left_cm: Optional[float], right_cm: Optional[float], *,
              nudge_cm: float, gain: float, turn_speed: float) -> float:
    bias = 0.0
    if left_cm is not None and left_cm < nudge_cm:
        bias -= gain * (1.0 - left_cm / nudge_cm)
    if right_cm is not None and right_cm < nudge_cm:
        bias += gain * (1.0 - right_cm / nudge_cm)
    return max(-1.0, min(1.0, bias)) * turn_speed


def compare_room(left_cm: float, right_cm: float, *, roomy_cm: float,
                  tie_margin_cm: float) -> Optional[bool]:
    """True = left has more room, False = right, None = tie (caller breaks it)."""
    near_left = min(left_cm, roomy_cm)
    near_right = min(right_cm, roomy_cm)
    if abs(near_left - near_right) < tie_margin_cm:
        return None
    return near_right < near_left
