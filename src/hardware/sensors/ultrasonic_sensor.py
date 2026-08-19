"""
Low-level HC-SR04 driver.

Responsibilities
----------------
- Configure GPIO.
- Trigger a measurement.
- Convert echo time into distance.
- Expose the latest measured distance.

No robot logic belongs here.
No emergency stop.
No grasp validation.
No wall avoidance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


SPEED_OF_SOUND_CM_PER_S = 34300.0
MIN_VALID_DISTANCE_CM = 2.0

# Median of the last N pings. A single dropped echo (common on HC-SR04) used
# to swing the reported distance by tens of cm, which downstream thresholds
# read as the obstacle appearing and vanishing every tick.
MEDIAN_WINDOW = 3


@dataclass(frozen=True)
class UltrasonicPins:
    trig: int
    echo: int


class UltrasonicSensor:

    def __init__(self, pins: UltrasonicPins):

        self._pins = pins
        self._distance_cm: Optional[float] = None
        self._history: deque = deque(maxlen=MEDIAN_WINDOW)

        if GPIO is None:
            return

        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self._pins.trig, GPIO.OUT)
        GPIO.setup(self._pins.echo, GPIO.IN)

        GPIO.output(self._pins.trig, False)

        time.sleep(0.05)

    def update(self) -> None:
        """
        Perform one ultrasonic measurement.

        Stores the latest measured distance internally.
        """

        if GPIO is None:
            self._record(None)
            return

        GPIO.output(self._pins.trig, True)
        time.sleep(0.00001)
        GPIO.output(self._pins.trig, False)

        timeout = time.monotonic() + 0.03

        while GPIO.input(self._pins.echo) == 0:
            if time.monotonic() > timeout:
                self._record(None)
                return

        pulse_start = time.monotonic()

        while GPIO.input(self._pins.echo) == 1:
            if time.monotonic() > timeout:
                self._record(None)
                return

        pulse_end = time.monotonic()

        pulse_duration = pulse_end - pulse_start

        distance_cm = pulse_duration * SPEED_OF_SOUND_CM_PER_S / 2.0
        self._record(distance_cm if distance_cm >= MIN_VALID_DISTANCE_CM else None)

    def _record(self, reading: Optional[float]) -> None:
        """Median-filter the raw ping. None (no echo) is kept in the window so
        a genuinely empty field of view still reports None once the window
        agrees -- a dropout alone can no longer move the reported distance."""
        self._history.append(reading)
        valid = sorted(r for r in self._history if r is not None)
        # Lower of the two middles on an even count: when the window is split,
        # report the nearer obstacle rather than the roomier one.
        self._distance_cm = valid[(len(valid) - 1) // 2] if len(valid) * 2 > len(self._history) else None

    def get_distance_cm(self) -> Optional[float]:
        """
        Return the latest measured distance.

        Returns
        -------
        float
            Latest distance in centimetres.

        None
            No valid reading.
        """
        return self._distance_cm

    def close(self) -> None:
        if GPIO is None:
            return

        # Some host environments provide a GPIO module but never had
        # `setmode()` called (or it was cleaned up elsewhere). Calling
        # `GPIO.cleanup()` in that state raises a RuntimeError:
        # "Please set pin numbering mode using GPIO.setmode(...)".
        # Guard by checking the current mode first where available.
        mode = None
        try:
            mode = GPIO.getmode()
        except Exception:
            # If getmode() is not available or errors, fall back to
            # attempting cleanup but swallow mode-check errors to avoid
            # masking the real intent — only call cleanup when mode set.
            mode = None

        if mode is not None:
            GPIO.cleanup([self._pins.trig, self._pins.echo])