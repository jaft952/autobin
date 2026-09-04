"""Shared pigpiod connection helper.

pigpiod timestamps GPIO edges and generates PWM in its own process, so
neither is disturbed when YOLO holds the GIL. Every module that wants it
asks here, and falls back to RPi.GPIO on its own when this returns None.

Each caller gets its OWN connection: the daemon supports many, and a
shared handle would let one module's close() cut another module's GPIO.
"""
from __future__ import annotations

from typing import Optional


def connect() -> Optional[object]:
    """A new pigpiod connection, or None (daemon down, not installed, or
    Pi 5, where pigpio is unsupported)."""
    try:
        import pigpio  # type: ignore
        pi = pigpio.pi()
        if pi.connected:
            return pi
    except Exception:
        pass
    return None
