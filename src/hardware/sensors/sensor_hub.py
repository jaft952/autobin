"""
src/hardware/sensors/sensor_hub.py

SensorHub — the single `sensors` object handed to every subsumption layer.

Composes the individual hardware sensors (camera/YOLO, ultrasonic) behind the
SensorInterface contract so layers stay hardware-free (Rule 2) and only ever
poll semantic getters (Rule 3). Any sensor can be omitted (None): its getters
then return the interface defaults, which lets tests run e.g. the zigzag scan
without the camera attached.

Up to 4 ultrasonics: front/left/right/back. front is the only one Layer 1
(scan) reads today, via get_obstacle_distance_cm() -- left/right/back exist
for all-round emergency coverage (has_obstacle()) and are otherwise exposed
read-only for future layers to use.

Obstacle semantics — two thresholds on purpose:
    get_obstacle_distance_cm()  raw filtered range, FRONT sensor only. Layer 1
                                (scan) uses this to trigger its zigzag
                                lane-turn EARLY (~35 cm).
    has_obstacle()              True when ANY fitted ultrasonic (front, left,
                                right, back) is INSIDE EMERGENCY_STOP_CM. This
                                is what Layer 5 (emergency stop) polls, so it
                                only fires if the scan layer failed to turn
                                away in time -- and now also catches a side or
                                rear collision the front sensor can't see.
Keeping the scan threshold well above the emergency threshold is what lets the
robot patrol without constantly tripping the emergency halt.
"""
from __future__ import annotations

from typing import Optional

from src.hardware.sensors.interfaces import SensorInterface


class SensorHub(SensorInterface):

    # Inside this range the situation is "imminent collision": Layer 5 halts.
    EMERGENCY_STOP_CM = 10.0

    def __init__(self, front=None, left=None, right=None, back=None, camera=None):
        """
        Args:
            front, left, right, back: UltrasonicSensor instances, or None if
                                       not fitted.
            camera:                   CameraSensor instance, or None if not
                                       fitted.
        """
        self._front = front
        self._left = left
        self._right = right
        self._back = back
        self._camera = camera
        self._ultrasonics = [s for s in (front, left, right, back) if s is not None]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Call once before the main loop."""
        if self._camera is not None:
            self._camera.start()

    def stop(self):
        """Call once on shutdown."""
        if self._camera is not None:
            self._camera.stop()
        for sensor in self._ultrasonics:
            sensor.close()

    # ── SensorInterface implementation ────────────────────────────────────

    def update(self):
        """Poll every fitted sensor. Called once per tick by the main loop."""
        if self._camera is not None:
            self._camera.update()
        for sensor in self._ultrasonics:
            sensor.update()

    def get_obstacle_distance_cm(self) -> Optional[float]:
        if self._front is None:
            return None
        return self._front.get_distance_cm()

    def get_obstacle_distance_left_cm(self) -> Optional[float]:
        return self._left.get_distance_cm() if self._left is not None else None

    def get_obstacle_distance_right_cm(self) -> Optional[float]:
        return self._right.get_distance_cm() if self._right is not None else None

    def get_obstacle_distance_back_cm(self) -> Optional[float]:
        return self._back.get_distance_cm() if self._back is not None else None

    def has_obstacle(self) -> bool:
        return any(
            (d := sensor.get_distance_cm()) is not None and d < self.EMERGENCY_STOP_CM
            for sensor in self._ultrasonics
        )

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
