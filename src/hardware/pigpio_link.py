"""Shared pigpiod connection helper; each caller gets its own connection so one module's close() can't cut another's GPIO."""
from __future__ import annotations

from typing import Optional


def connect() -> Optional[object]:
    """A new pigpiod connection, or None (daemon down, not installed, or Pi 5)."""
    try:
        import pigpio  # type: ignore
        pi = pigpio.pi()
        if pi.connected:
            return pi
    except Exception:
        pass
    return None
