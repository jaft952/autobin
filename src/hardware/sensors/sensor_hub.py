"""
src/hardware/sensors/sensor_hub.py

SensorHub — the single `sensors` object handed to every subsumption layer.

Composes the individual hardware sensors (camera/YOLO, ultrasonic) behind the
SensorInterface contract so layers stay hardware-free (Rule 2) and only ever
poll semantic getters (Rule 3). Any sensor can be omitted (None): its getters
then return the interface defaults, which lets tests run e.g. the zigzag scan
without the camera attached.

Obstacle semantics — two thresholds on purpose:
    get_obstacle_distance_cm()  raw filtered range. Layer 1 (scan) uses this to
                                trigger its zigzag lane-turn EARLY (~35 cm).
    has_obstacle()              True only when an obstacle is INSIDE
                                EMERGENCY_STOP_CM. This is what Layer 5
                                (emergency stop) polls, so it only fires if the
                                scan layer failed to turn away in time.
Keeping the scan threshold well above the emergency threshold is what lets the
robot patrol without constantly tripping the emergency halt.
"""
from __future__ import annotations

from typing import Optional

from src.hardware.sensors.interfaces import SensorInterface


class SensorHub(SensorInterface):

    # Inside this range the situation is "imminent collision": Layer 5 halts.
    EMERGENCY_STOP_CM = 10.0

    def __init__(self, ultrasonic=None, camera=None):
        """
        Args:
            ultrasonic: UltrasonicSensor instance, or None if not fitted.
            camera:     CameraSensor instance, or None if not fitted.
        """
        self._ultrasonic = ultrasonic
        self._camera = camera

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Call once before the main loop."""
        if self._camera is not None:
            self._camera.start()

    def stop(self):
        """Call once on shutdown."""
        if self._camera is not None:
            self._camera.stop()
        if self._ultrasonic is not None:
            self._ultrasonic.close()

    # ── SensorInterface implementation ────────────────────────────────────

    def update(self):
        """Poll every fitted sensor. Called once per tick by the main loop."""
        if self._camera is not None:
            self._camera.update()
        if self._ultrasonic is not None:
            self._ultrasonic.update()

    def get_obstacle_distance_cm(self) -> Optional[float]:
        if self._ultrasonic is None:
            return None
        return self._ultrasonic.get_distance_cm()

    def has_obstacle(self) -> bool:
        dist = self.get_obstacle_distance_cm()
        return dist is not None and dist < self.EMERGENCY_STOP_CM

    def get_litter_position(self):
        if self._camera is None:
            return None
        return self._camera.get_litter_position()

    def get_litter_ground_contact(self):
        if self._camera is None:
            return None
        return self._camera.get_litter_ground_contact()

    def get_litter_pose(self):
        if self._camera is None:
            return None
        return self._camera.get_litter_pose()

    def get_aerial_trash_position(self):
        if self._camera is None:
            return None
        return self._camera.get_aerial_trash_position()

    def get_battery_level(self) -> float:
        if self._camera is not None:
            return self._camera.get_battery_level()
        return 1.0

    # ── Debug passthrough ─────────────────────────────────────────────────

    def get_annotated_frame(self):
        if self._camera is None:
            return None
        return self._camera.get_annotated_frame()
