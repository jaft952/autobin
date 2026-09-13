"""Camera sensor that runs YOLO to find cans."""

import math
import threading
import time

from src.hardware.sensors.interfaces import SensorInterface
from src.perception.detector import (
    AluminiumCanDetector, DetectionResult, RUNTIME_MODEL_PATH,
)
from src.perception.target_lock import TargetLock

STALE_AFTER_S = 1.5


class CameraSensor(SensorInterface):
    """Webcam plus YOLO11n-seg can detection."""

    def __init__(
        self,
        model_path: str = RUNTIME_MODEL_PATH,
        camera_index: int = 0,
        conf_threshold: float = 0.75,
        frame_width: int = 1280,
        frame_height: int = 720,
        device: str = "cpu",
    ):
        self._detector = AluminiumCanDetector(
            model_path=model_path,
            camera_index=camera_index,
            conf_threshold=conf_threshold,
            frame_width=frame_width,
            frame_height=frame_height,
            device=device,
        )
        self._latest_result: DetectionResult = DetectionResult()
        self._latest_at: float = 0.0
        self._result_lock = threading.Lock()
        self._target_lock = TargetLock()
        self._selected_box = None
        self._selected_at: float = -1.0
        self._alive = False
        self._worker = None
        self._running = threading.Event()
        self._running.set()
        self._camera_enabled = True

    def start(self):
        """Start the camera and the detection thread."""
        self._detector.start()
        self._alive = True
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="yolo-worker")
        self._worker.start()

    def stop(self):
        """Stop detection and release the camera."""
        self._alive = False
        self._running.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._detector.stop()

    def pause(self):
        """Pause detection until resume()."""
        self._running.clear()

    def resume(self):
        if self._camera_enabled:
            self._running.set()

    def camera_off(self) -> None:
        """Turn the camera off to save battery."""
        self._camera_enabled = False
        self.pause()
        self._detector.stop()

    def camera_on(self) -> None:
        """Turn the camera back on and resume detection."""
        self._detector.reopen_camera()
        self._camera_enabled = True
        self.resume()

    @property
    def camera_enabled(self) -> bool:
        return self._camera_enabled

    def wait_for_fresh_frames(self, n: int = 2, timeout: float = 2.0) -> int:
        """Wait until n new detection results are ready."""
        start_at = self._latest_at
        deadline = time.monotonic() + timeout
        seen = 0
        last_at = start_at
        while seen < n and time.monotonic() < deadline:
            time.sleep(0.02)
            at = self._latest_at
            if at != last_at:
                seen += 1
                last_at = at
        return seen

    def _run(self):
        """Capture and detect in a loop."""
        while self._alive:
            if not self._running.wait(timeout=0.2):
                continue
            try:
                result = self._detector.detect()
            except Exception as exc:
                print(f"[CameraSensor] inference error: {exc!r}")
                time.sleep(0.5)
                continue
            with self._result_lock:
                self._latest_result = result
                self._latest_at = time.monotonic()

    def update(self):
        """Nothing heavy here, detection runs in its own thread."""
        pass

    def _current_target(self):
        """Box and result of the locked can."""
        with self._result_lock:
            result = self._latest_result
            at = self._latest_at
            stale = time.monotonic() - at > STALE_AFTER_S
        if stale:
            return None, result
        if at != self._selected_at:
            self._selected_at = at
            self._selected_box = self._target_lock.select(result.detections)
        return self._selected_box, result

    def release_target(self):
        """Forget the locked can."""
        self._target_lock.release()
        self._selected_box = None
        self._selected_at = -1.0

    def get_litter_position(self):  # type: ignore
        """Image position of the locked can, or None."""
        box, result = self._current_target()
        if box is None or result.frame_width == 0:
            return None
        return (box.center_x / result.frame_width,
                box.center_y / result.frame_height)

    def get_litter_locked(self) -> bool:  # type: ignore
        """True while a can is locked."""
        return self._target_lock.locked

    def get_litter_ground_contact(self):  # type: ignore
        """Point where the locked can touches the floor."""
        box, result = self._current_target()
        if box is None or result.frame_width == 0:
            return None
        u, v = box.base_center
        return (u / result.frame_width, v / result.frame_height)

    def get_litter_distance_cm(self):  # type: ignore
        box, result = self._current_target()
        if box is None or box.width <= 0 or box.height <= 0:
            return None
        from src.visual_servoing.distance_error import CALIBRATION_CONSTANT_PX_CM
        bbox_area_px = box.width * box.height
        return math.sqrt(CALIBRATION_CONSTANT_PX_CM / bbox_area_px)

    def get_litter_too_close(self):  # type: ignore
        box, result = self._current_target()
        if box is None or result.frame_height == 0:
            return False
        from src.visual_servoing.distance_error import (
            CLOSE_BBOX_FRACTION, TOO_CLOSE_DISTANCE_CM)
        if box.height / result.frame_height >= CLOSE_BBOX_FRACTION:
            return True
        distance_cm = self.get_litter_distance_cm()
        return distance_cm is not None and distance_cm <= TOO_CLOSE_DISTANCE_CM

    def get_litter_target_error(self):  # type: ignore
        """Steering error to the locked can."""
        box, result = self._current_target()
        from src.visual_servoing.distance_error import compute_target_error
        locked_result = DetectionResult(
            detections=[box] if box is not None else [],
            frame_width=result.frame_width, frame_height=result.frame_height)
        return compute_target_error(locked_result)

    def get_litter_pose(self):  # type: ignore
        """Pose of the locked can from its mask."""
        box, _ = self._current_target()
        if box is None or box.orientation is None:
            return None
        o = box.orientation
        return {"klass": o.klass, "angle": o.angle}

    def get_aerial_trash_position(self):  # type: ignore
        """Not used."""
        return None

    def has_obstacle(self) -> bool:
        """Not used by the camera."""
        return False

    def get_latest_result(self) -> DetectionResult:
        """Newest detection result, for debugging."""
        with self._result_lock:
            return self._latest_result

    def get_annotated_frame(self):
        """Last frame with detection boxes drawn."""
        return self._detector.get_annotated_frame(self.get_latest_result(), target=self._target_lock.target)

    def read_camera_frame(self):
        """Read one raw camera frame without detection."""
        if not self._detector._cap or not self._detector._cap.isOpened():
            self._detector._open_camera()
        return self._detector.read_frame()
