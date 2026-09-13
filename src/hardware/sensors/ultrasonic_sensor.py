"""HC-SR04 ultrasonic sensor driver."""

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

ECHO_TIMEOUT_S = 0.010

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
        self._lock = threading.Lock()

        if GPIO is None:
            return

        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self._pins.trig, GPIO.OUT)
        GPIO.setup(self._pins.echo, GPIO.IN)

        GPIO.output(self._pins.trig, False)

        time.sleep(0.05)

    def update(self) -> None:
        """Take one distance reading."""

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
        """Store a reading and filter out single dropped echoes."""
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

        mode = None
        try:
            mode = GPIO.getmode()
        except Exception:
            mode = None

        if mode is not None:
            GPIO.cleanup([self._pins.trig, self._pins.echo])