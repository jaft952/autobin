"""CameraSensor: SensorInterface impl. YOLO runs in a bg thread so the control loop stays fast; getters read the latest locked-tin result."""

import math
import threading
import time

from src.hardware.sensors.interfaces import LitterSnapshot, SensorInterface
from src.perception.detector import (
    AluminiumCanDetector, DetectionResult, RUNTIME_MODEL_PATH,
)
from src.perception.target_lock import TargetLock

# Result older than this counts as stale, not real.
STALE_AFTER_S = 1.5


class CameraSensor(SensorInterface):
    """Webcam + YOLO11n-seg sensor exposing locked-tin position/pose to the Subsumption layers."""

    def __init__(
        self,
        model_path: str = RUNTIME_MODEL_PATH,   # single source of truth
        camera_index: int = 0,
        conf_threshold: float = 0.75,   # keep = AluminiumCanDetector's default
        frame_width: int = 1280,
        frame_height: int = 720,
        device: str = "cpu",
    ):
        # forced cpu: this Pi's CUDA torch build SIGILLs on first real op
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
        self._running.set()          # cleared only while the arm is grabbing
        self._camera_enabled = True  # dashboard battery-save toggle

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Start the camera and the background inference worker."""
        self._detector.start()
        self._alive = True
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="yolo-worker")
        self._worker.start()

    def stop(self):
        """Stop the worker and release the camera."""
        self._alive = False
        self._running.set()          # never leave the worker parked on the wait
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._detector.stop()

    def pause(self):
        """Stop inferring until resume()."""
        self._running.clear()

    def resume(self):
        if self._camera_enabled:   # don't let a grab's auto-resume override a manual camera_off()
            self._running.set()

    def camera_off(self) -> None:
        """Release the camera hardware (battery-save toggle)."""
        self._camera_enabled = False
        self.pause()
        self._detector.stop()

    def camera_on(self) -> None:
        """Reopen the camera hardware and resume inference."""
        self._detector.reopen_camera()
        self._camera_enabled = True
        self.resume()

    @property
    def camera_enabled(self) -> bool:
        return self._camera_enabled

    def wait_for_fresh_frames(self, n: int = 2, timeout: float = 2.0) -> int:
        """Block until `n` new inference results are published. Returns how many were seen."""
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
        """Capture and infer continuously; publish the latest result."""
        while self._alive:
            # timed wait so stop() during a pause still ends the thread
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

    # ── SensorInterface Implementation ─────────────────────────────────

    def update(self):
        """No-op; inference runs in the worker thread."""
        pass

    def _current_target(self):
        """(box, result) of the locked tin, selected once per new frame."""
        with self._result_lock:
            result = self._latest_result
            at = self._latest_at
            if time.monotonic() - at > STALE_AFTER_S:
                return None, result
            if at != self._selected_at:
                self._selected_at = at
                self._selected_box = self._target_lock.select(result.detections)
            return self._selected_box, result

    def snapshot(self) -> LitterSnapshot:
        """All locked-tin facts from one frame, so callers can't straddle two frames."""
        box, result = self._current_target()
        if box is None or result.frame_width == 0 or result.frame_height == 0:
            return LitterSnapshot(None, None, None, None)
        o = box.orientation
        u, v = box.base_center
        return LitterSnapshot(
            center=(box.center_x / result.frame_width,
                    box.center_y / result.frame_height),
            ground_contact=(u / result.frame_width, v / result.frame_height),
            klass=o.klass if o is not None else None,
            angle=o.angle if o is not None else None,
        )

    def release_target(self):
        """Forget the current tin so the next frame re-picks the largest."""
        self._target_lock.release()
        self._selected_box = None
        self._selected_at = -1.0

    def get_litter_position(self): # type: ignore
        """Normalized (x, y) of the locked can, or None."""
        box, result = self._current_target()
        if box is None or result.frame_width == 0:
            return None
        return (box.center_x / result.frame_width,
                box.center_y / result.frame_height)

    def get_litter_locked(self) -> bool: # type: ignore
        """True while TargetLock holds a tin."""
        return self._target_lock.locked

    def get_litter_ground_contact(self): # type: ignore
        """Normalized (x, y) of the locked tin's ground-contact point, or None."""
        box, result = self._current_target()
        if box is None or result.frame_width == 0:
            return None
        u, v = box.base_center
        return (u / result.frame_width, v / result.frame_height)

    def get_litter_distance_cm(self): # type: ignore
        box, result = self._current_target()
        if box is None or box.width <= 0 or box.height <= 0:
            return None
        from src.visual_servoing.distance_error import CALIBRATION_CONSTANT_PX_CM
        bbox_area_px = box.width * box.height
        return math.sqrt(CALIBRATION_CONSTANT_PX_CM / bbox_area_px)

    def get_litter_too_close(self): # type: ignore
        box, result = self._current_target()
        if box is None or result.frame_height == 0:
            return False
        from src.visual_servoing.distance_error import (
            CLOSE_BBOX_FRACTION, TOO_CLOSE_DISTANCE_CM)
        if box.height / result.frame_height >= CLOSE_BBOX_FRACTION:
            return True
        distance_cm = self.get_litter_distance_cm()
        return distance_cm is not None and distance_cm <= TOO_CLOSE_DISTANCE_CM

    def get_litter_target_error(self): # type: ignore
        """Same TargetError the live IBVS test loop computes, for the locked target."""
        box, result = self._current_target()
        from src.visual_servoing.distance_error import compute_target_error
        locked_result = DetectionResult(
            detections=[box] if box is not None else [],
            frame_width=result.frame_width, frame_height=result.frame_height)
        return compute_target_error(locked_result)

    def get_litter_pose(self): # type: ignore
        """Locked tin's pose from its mask: {'klass', 'angle'}, or None."""
        box, _ = self._current_target()
        if box is None or box.orientation is None:
            return None
        o = box.orientation
        return {"klass": o.klass, "angle": o.angle}

    def get_aerial_trash_position(self): # type: ignore
        """Not used for floor litter. Returns None."""
        return None

    def has_obstacle(self) -> bool:
        """Placeholder. Replace with actual proximity sensor reading."""
        return False

    # ── Extra: Access raw result for debugging ──────────────────────────

    def get_latest_result(self) -> DetectionResult:
        """Newest DetectionResult (even if stale), for debugging."""
        with self._result_lock:
            return self._latest_result

    def get_annotated_frame(self):
        """Last frame with boxes drawn and the locked target highlighted."""
        return self._detector.get_annotated_frame(self.get_latest_result(), target=self._target_lock.target)

    def read_camera_frame(self):
        """One raw camera frame, no model load or inference."""
        if not self._detector._cap or not self._detector._cap.isOpened():
            self._detector._open_camera()
        return self._detector.read_frame()
