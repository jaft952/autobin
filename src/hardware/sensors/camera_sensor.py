"""
src/hardware/sensors/camera_sensor.py

CameraSensor — Concrete implementation of SensorInterface.
Bridges AluminiumCanDetector (perception) with the Subsumption layers.

THREADING (2026-07-08): YOLO inference runs in its OWN background thread,
not inside update(). On the Pi's CPU one inference takes hundreds of ms; if
update() ran it synchronously the whole 10 Hz control loop (ultrasonic
avoidance, emergency stop, zigzag timing) would be dragged down to 2-3 Hz.
Instead the worker continuously captures + infers at whatever rate the model
manages, and the polled getters read the LATEST result — the control loop
stays at full rate and simply sees detections a frame-age late.

Staleness guard: if the worker hasn't produced a result recently (camera
unplugged, inference crash-looping), the getters report "no detection"
rather than acting on a frozen frame.

RESOLUTION: capture defaults to 1280x720 (not 1920x1080). Inference is
letterboxed to 640 px anyway, so 1080p only added USB/decode/resize cost.
All downstream consumers use NORMALIZED coordinates, which are unchanged as
long as the camera's field of view is the same at both resolutions — verify
once on the Pi (arc ny values should match the calibration); if the FOV
differs, pass frame_width=1920, frame_height=1080 here instead of recalibrating.
"""

import threading
import time

from src.hardware.sensors.interfaces import SensorInterface
from src.perception.detector import (
    AluminiumCanDetector, DetectionResult, RUNTIME_MODEL_PATH,
)

# If the newest inference result is older than this, report "no detection"
# instead of acting on a frozen scene.
STALE_AFTER_S = 1.5


class CameraSensor(SensorInterface):
    """
    Concrete sensor that uses the webcam + YOLO11n-seg to detect aluminium
    cans and expose their position/pose to the Subsumption layers.

    Implements the SensorInterface contract:
        update()               — cheap; inference runs in a background thread
        get_litter_position()  — returns (norm_x, norm_y) or None
    """

    def __init__(
        self,
        model_path: str = RUNTIME_MODEL_PATH,   # single source of truth
        camera_index: int = 0,
        conf_threshold: float = 0.5,
        frame_width: int = 1280,
        frame_height: int = 720,
    ):
        self._detector = AluminiumCanDetector(
            model_path=model_path,
            camera_index=camera_index,
            conf_threshold=conf_threshold,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self._latest_result: DetectionResult = DetectionResult()
        self._latest_at: float = 0.0
        self._result_lock = threading.Lock()
        self._alive = False
        self._worker = None
        self._battery: float = 1.0   # Placeholder; replace with real battery sensor

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Start the camera and the background inference worker."""
        self._detector.start()
        self._alive = True
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="yolo-worker")
        self._worker.start()

    def stop(self):
        """Stop the worker and release the camera. Call on system shutdown."""
        self._alive = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._detector.stop()

    def _run(self):
        """Capture + infer as fast as the model allows; publish the result."""
        while self._alive:
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
        """Cheap by design — inference happens in the worker thread. Kept so
        the polled-sensor contract (update every tick) stays uniform."""
        pass

    def _fresh_result(self) -> DetectionResult:
        """Latest result, or an empty one when it has gone stale."""
        with self._result_lock:
            if time.monotonic() - self._latest_at > STALE_AFTER_S:
                return DetectionResult()
            return self._latest_result

    def get_litter_position(self):
        """
        Returns normalized (x, y) of the best-detected aluminium can,
        where (0.5, 0.5) is the center of the frame.
        Returns None if no can is detected (or detection has gone stale).

        Used by:
            layer1_scan.py    — to check if a target exists
            layer2_approach.py — to calculate motion vector towards the can
        """
        return self._fresh_result().normalized_center()

    def get_litter_ground_contact(self):
        """
        Returns normalized (x, y) of the best detection's ground-contact
        point (bbox bottom-center), or None. Arc-grasp and pixel_to_arm
        calibrations are anchored to this point.

        Used by:
            layer3_collect.py — to solve the grasp pose
        """
        return self._fresh_result().normalized_base_center()

    def get_litter_pose(self):
        """
        Returns the best detection's pose estimated from its segmentation
        mask: {'klass': 'upright'|'lying'|'axial', 'angle': deg 0..180},
        or None when there is no detection / no usable mask.

        Used by:
            layer3_collect.py — upright vs lying picks the grasp grid, and
            the angle drives the wrist roll (CH5) for lying tins.
        """
        best = self._fresh_result().best
        if best is None or best.orientation is None:
            return None
        o = best.orientation
        return {"klass": o.klass, "angle": o.angle}

    def get_aerial_trash_position(self):
        """Not used for floor litter. Returns None."""
        return None

    def get_battery_level(self) -> float:
        """Placeholder. Replace with actual battery sensor reading."""
        return self._battery

    def has_obstacle(self) -> bool:
        """Placeholder. Replace with actual proximity sensor reading."""
        return False

    # ── Extra: Access raw result for debugging ──────────────────────────

    def get_latest_result(self) -> DetectionResult:
        """Returns the newest DetectionResult (even if stale) for debugging."""
        with self._result_lock:
            return self._latest_result

    def get_annotated_frame(self):
        """
        Returns the last camera frame with bounding boxes drawn.
        Use with cv2.imshow() or the web dashboard's MJPEG stream.
        """
        return self._detector.get_annotated_frame(self.get_latest_result())
