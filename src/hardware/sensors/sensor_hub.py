"""One object that holds all sensors for the layers."""
from __future__ import annotations

from typing import Optional

from src.hardware.sensors.interfaces import SensorInterface
from src.hardware.sensors.ultrasonic_array import UltrasonicArray


class SensorHub(SensorInterface):

    EMERGENCY_STOP_CM = 15.0

    def __init__(self, front=None, back=None, front_left=None, front_right=None,
                 camera=None):
        """Any sensor can be None if it is not fitted."""
        self._front = front
        self._back = back
        self._front_left = front_left
        self._front_right = front_right
        self._camera = camera
        self._ultrasonics = [s for s in (front, back, front_left, front_right) if s is not None]
        self._ultrasonic_array = UltrasonicArray(self._ultrasonics)

    def start(self):
        """Start all sensors. Call once before the main loop."""
        if self._camera is not None:
            self._camera.start()
        self._ultrasonic_array.start()

    def stop(self):
        """Stop all sensors. Call once on shutdown."""
        if self._camera is not None:
            self._camera.stop()
        self._ultrasonic_array.stop()

    def update(self):
        """Update the camera. Ultrasonics update on their own thread."""
        if self._camera is not None:
            self._camera.update()

    def get_obstacle_distance_cm(self) -> Optional[float]:  # type: ignore
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

    def get_litter_position(self):  # type: ignore
        if self._camera is None:
            return None
        return self._camera.get_litter_position()

    def get_litter_locked(self) -> bool:
        if self._camera is None:
            return False
        return self._camera.get_litter_locked()

    def get_litter_ground_contact(self):  # type: ignore
        if self._camera is None:
            return None
        return self._camera.get_litter_ground_contact()

    def get_litter_pose(self):  # type: ignore
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

    def get_litter_target_error(self):
        if self._camera is None:
            return None
        return self._camera.get_litter_target_error()

    def get_aerial_trash_position(self):  # type: ignore
        if self._camera is None:
            return None
        return self._camera.get_aerial_trash_position()

    def pause_camera(self):
        """Pause detection so the arm gets more CPU."""
        if self._camera is not None and hasattr(self._camera, "pause"):
            self._camera.pause()

    def resume_camera(self):
        if self._camera is not None and hasattr(self._camera, "resume"):
            self._camera.resume()

    def wait_for_fresh_frames(self, n: int = 2, timeout: float = 2.0) -> int:
        if self._camera is not None and hasattr(self._camera, "wait_for_fresh_frames"):
            return self._camera.wait_for_fresh_frames(n, timeout)
        return 0

    def release_target(self):
        """Forget the locked can after a pickup."""
        if self._camera is not None and hasattr(self._camera, "release_target"):
            self._camera.release_target()

    def camera_off(self) -> None:
        """Turn the camera off to save battery."""
        if self._camera is not None and hasattr(self._camera, "camera_off"):
            self._camera.camera_off()

    def camera_on(self) -> None:
        if self._camera is not None and hasattr(self._camera, "camera_on"):
            self._camera.camera_on()

    @property
    def camera_enabled(self) -> bool:
        if self._camera is None:
            return False
        return getattr(self._camera, "camera_enabled", True)

    def get_annotated_frame(self):
        if self._camera is None:
            return None
        return self._camera.get_annotated_frame()
