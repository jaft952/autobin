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

from dataclasses import dataclass
from typing import Optional
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


SPEED_OF_SOUND_CM_PER_S = 34300.0


@dataclass(frozen=True)
class UltrasonicPins:
    trig: int
    echo: int


class UltrasonicSensor:

    def __init__(self, pins: UltrasonicPins):

        self._pins = pins
        self._distance_cm: Optional[float] = None

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
            self._distance_cm = None
            return

        GPIO.output(self._pins.trig, True)
        time.sleep(0.00001)
        GPIO.output(self._pins.trig, False)

        timeout = time.monotonic() + 0.03

        while GPIO.input(self._pins.echo) == 0:
            if time.monotonic() > timeout:
                self._distance_cm = None
                return

        pulse_start = time.monotonic()

        while GPIO.input(self._pins.echo) == 1:
            if time.monotonic() > timeout:
                self._distance_cm = None
                return

        pulse_end = time.monotonic()

        pulse_duration = pulse_end - pulse_start

        self._distance_cm = (
            pulse_duration * SPEED_OF_SOUND_CM_PER_S / 2.0
        )

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
        if GPIO is not None:
            GPIO.cleanup([self._pins.trig, self._pins.echo])