"""
High-level safety logic for the front ultrasonic sensor.

Responsibilities
----------------
1. Poll the HC-SR04.
2. Decide whether emergency stop is required.
3. Decide whether the arm is allowed to grab.
4. Never contains GPIO code other than calling UltrasonicSensor.

Pure decision layer built on top of the hardware driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.hardware.sensors.ultrasonic_sensor import (
    UltrasonicSensor,
    UltrasonicPins,
)

EMERGENCY_STOP_CM = 15.0
GRAB_CONFIRM_CM = 25.0


@dataclass
class UltrasonicState:
    distance_cm: Optional[float]
    emergency_stop: bool
    grab_confirmed: bool


class UltrasonicSafety:

    def __init__(
        self,
        trig: int = 23,
        echo: int = 24,
    ):
        self.sensor = UltrasonicSensor(
            UltrasonicPins(trig=trig, echo=echo)
        )

    def update(self) -> UltrasonicState:
        """
        Poll the sensor once and return the current safety state.
        """

        self.sensor.update()

        d = self.sensor.get_distance_cm()

        if d is None:
            return UltrasonicState(
                distance_cm=None,
                emergency_stop=False,
                grab_confirmed=False,
            )

        return UltrasonicState(
            distance_cm=d,
            emergency_stop=d <= EMERGENCY_STOP_CM,
            grab_confirmed=d <= GRAB_CONFIRM_CM,
        )

    def should_stop(self) -> bool:
        """
        Convenience method.
        """
        s = self.update()
        return s.emergency_stop

    def can_grab(self) -> bool:
        """
        Convenience method.
        """
        s = self.update()
        return s.grab_confirmed

    def close(self):
        self.sensor.close()