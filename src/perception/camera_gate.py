"""Pure decision for whether the camera should be woken by proximity.

No hardware access, no state: the caller (RobotRuntime) owns the actual
camera_on()/camera_off() calls, so this stays unit-testable with plain
numbers, same pattern as src/safety/obstacle_avoidance.py.
"""
from __future__ import annotations

from typing import Optional

# Read qualified (camera_gate.WAKE_DISTANCE_CM) elsewhere so the dashboard
# can monkeypatch this one value, same convention as src/scanning/tuning.py.
WAKE_DISTANCE_CM = 50.0


def should_wake_camera(front: Optional[float], left: Optional[float],
                        right: Optional[float],
                        wake_cm: float = WAKE_DISTANCE_CM) -> bool:
    """True if any forward-facing ultrasonic (front, front-left, front-right)
    reads within wake_cm. None (no echo) never wakes the camera on its own."""
    return any(d is not None and d <= wake_cm for d in (front, left, right))
