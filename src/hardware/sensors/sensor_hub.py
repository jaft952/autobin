"""
src/hardware/sensors/sensor_hub.py

SensorHub — the single `sensors` object handed to every subsumption layer.

Composes the individual hardware sensors (camera/YOLO, ultrasonic) behind the
SensorInterface contract so layers stay hardware-free (Rule 2) and only ever
poll semantic getters (Rule 3). Any sensor can be omitted (None): its getters
then return the interface defaults, which lets tests run e.g. the zigzag scan
without the camera attached.

Up to 4 ultrasonics: front / back / front_left / front_right (the last two
are diagonals). They are pinged ROUND-ROBIN, one per tick: firing them
back-to-back let each receiver hear its neighbour's outgoing burst directly
through the air, which decodes as a phantom obstacle a few cm away. One tick
apart clears the HC-SR04 datasheet's ~60ms between-measurement minimum.

Obstacle semantics — two thresholds on purpose:
    get_obstacle_distance_cm()  FRONT sensor only. Layer 1 (scan) turns its
                                zigzag lane EARLY on this (~35 cm).
    has_obstacle()              True when ANY fitted ultrasonic is inside
                                EMERGENCY_STOP_CM. A coarse summary only:
                                the layers read the per-direction getters,
                                because the rear must be acted on solely
                                while reversing.
Keeping the scan threshold well above the emergency threshold is what lets the
robot patrol without constantly tripping the emergency halt.
"""
from __future__ import annotations

from typing import Optional

from src.hardware.sensors.interfaces import SensorInterface


class SensorHub(SensorInterface):

    # Inside this range the situation is "imminent collision": Layer 5 halts.
    EMERGENCY_STOP_CM = 15.0

    def __init__(self, front=None, back=None, front_left=None, front_right=None,
                 camera=None, battery=None):
        """Any sensor may be None if not fitted."""
        self._front = front
        self._back = back
        self._front_left = front_left
        self._front_right = front_right
        self._camera = camera
        self._battery = battery
        self._ultrasonics = [s for s in (front, back, front_left, front_right) if s is not None]
        self._next_ping = 0

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
        if self._ultrasonics:
            self._ultrasonics[self._next_ping].update()
            self._next_ping = (self._next_ping + 1) % len(self._ultrasonics)

    def get_obstacle_distance_cm(self) -> Optional[float]: # type: ignore
        if self._front is None:
            return None
        return self._front.get_distance_cm()

    def get_obstacle_distance_back_cm(self) -> Optional[float]:  # type: ignore
        return self._back.get_distance_cm() if self._back is not None else None

    def get_obstacle_distance_front_left_cm(self) -> Optional[float]:  # type: ignore
        return self._front_left.get_distance_cm() if self._front_left is not None else None

    def get_obstacle_distance_front_right_cm(self) -> Optional[float]:  # type: ignore
        return self._front_right.get_distance_cm() if self._front_right is not None else None

    def has_obstacle(self) -> bool:
        return any(
            (d := sensor.get_distance_cm()) is not None and d < self.EMERGENCY_STOP_CM
            for sensor in self._ultrasonics
        )

    def get_litter_position(self): # type: ignore
        if self._camera is None:
            return None
        return self._camera.get_litter_position()

    def get_litter_ground_contact(self): # type: ignore
        if self._camera is None:
            return None
        return self._camera.get_litter_ground_contact()

    def get_litter_pose(self): # type: ignore
        if self._camera is None:
            return None
        return self._camera.get_litter_pose()

    def get_litter_distance_cm(self):
        if self._camera is None:
            return None
        return self._camera.get_litter_distance_cm()

    def get_litter_too_close(self):
        if self._camera is None:
            return False
        return self._camera.get_litter_too_close()

    def get_aerial_trash_position(self): # type: ignore
        if self._camera is None:
            return None
        return self._camera.get_aerial_trash_position()

    def get_battery_level(self) -> float:
        if self._battery is not None:
            return self._battery.get_battery_level()
        if self._camera is not None:
            return self._camera.get_battery_level()
        return 1.0

    def has_real_battery_sensor(self) -> bool:
        return self._battery is not None

    # ── Camera worker control ─────────────────────────────────────────────

    def pause_camera(self):
        """Stop YOLO inference (it competes with the arm's move pacing for
        CPU — see CameraSensor's PAUSING note). No-op without a camera."""
        if self._camera is not None and hasattr(self._camera, "pause"):
            self._camera.pause()

    def resume_camera(self):
        if self._camera is not None and hasattr(self._camera, "resume"):
            self._camera.resume()

    def release_target(self):
        """Forget the locked tin (call after a collection)."""
        if self._camera is not None and hasattr(self._camera, "release_target"):
            self._camera.release_target()

    # ── Debug passthrough ─────────────────────────────────────────────────

    def get_annotated_frame(self):
        if self._camera is None:
            return None
        return self._camera.get_annotated_frame()
