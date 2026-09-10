"""Low-level HC-SR04 driver: GPIO trigger/echo -> distance. No robot logic here."""

from __future__ import annotations

import threading
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

# Longest echo worth waiting for. Range is timeout * SPEED_OF_SOUND / 2, so
# 0.010 s covers ~170 cm -- well past anything this robot acts on, and a
# third of the 0.03 s a missing echo used to burn before giving up. Only the
# no-echo case changes; a ping that answers is unaffected.
ECHO_TIMEOUT_S = 0.010

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
        self._lock = threading.Lock()   # update() runs on UltrasonicArray's thread, get_distance_cm() on the control loop

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

        timeout = time.monotonic() + ECHO_TIMEOUT_S

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
        """Nearest-of-window filter: None (no echo) stays in the window so a
        dropout alone can't move the reading, but the reported distance is
        the CLOSEST valid ping in the window, not the median. This is a
        collision sensor: under-reporting how close something is would delay
        an emergency stop until a second consecutive close ping arrived (up
        to ~2 ping cycles late on a fast-closing obstacle), while
        over-reporting it from one stray echo only costs an extra defensive
        backoff. Still requires a majority of the window to be a real echo
        before trusting any of it."""
        with self._lock:
            self._history.append(reading)
            valid = [r for r in self._history if r is not None]
            self._distance_cm = (min(valid)
                                  if len(valid) * 2 > len(self._history) else None)

    def get_distance_cm(self) -> Optional[float]:
        with self._lock:
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