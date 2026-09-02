"""Shared ultrasonic clearance reading: missing getter or no echo both mean 'no info', never 0cm."""
from __future__ import annotations

from typing import Any, NamedTuple, Optional

GETTER_FRONT = "get_obstacle_distance_cm"
GETTER_BACK = "get_obstacle_distance_back_cm"
GETTER_FRONT_LEFT = "get_obstacle_distance_front_left_cm"
GETTER_FRONT_RIGHT = "get_obstacle_distance_front_right_cm"


class Clearances(NamedTuple):
    front: Optional[float]
    back: Optional[float]
    front_left: Optional[float]
    front_right: Optional[float]


def read_distance_cm(sensors: Any, getter_name: str) -> Optional[float]:
    getter = getattr(sensors, getter_name, None)
    return getter() if getter is not None else None


def read_clearances(sensors: Any) -> Clearances:
    return Clearances(
        front=read_distance_cm(sensors, GETTER_FRONT),
        back=read_distance_cm(sensors, GETTER_BACK),
        front_left=read_distance_cm(sensors, GETTER_FRONT_LEFT),
        front_right=read_distance_cm(sensors, GETTER_FRONT_RIGHT),
    )


def clearance_cm(sensors: Any, getter_name: str, *,
                  default: float, cap: Optional[float] = None) -> float:
    """Cap applies to a present reading too, not just the default."""
    value = read_distance_cm(sensors, getter_name)
    result = default if value is None else value
    return min(result, cap) if cap is not None else result
