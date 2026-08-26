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

TARGET PRIORITY: the getters do NOT expose every detection — TargetLock
picks the largest-bbox tin, holds it across frames, and hides the rest until
that tin is collected (see src/perception/target_lock.py). One tin at a
time, nearest first, no re-aiming mid-approach.

PAUSING: the worker is paused around a grab (see RobotRuntime._tick). YOLO
saturates the Pi's CPU, and the arm's stepped_move paces itself against the
wall clock at 50 Hz -- a GIL stall longer than one 20 ms tick turns a smooth
descent into the stop-start jerking the stepped-move rewrite existed to
remove. Inference during a grab is worthless anyway: the arm occludes the
camera. tests/test_ibvs_centering.py gets this for free, because it infers on
its main loop, which is blocked for the whole grab.

RESOLUTION: capture defaults to 1280x720 (not 1920x1080). Inference is
letterboxed to 640 px anyway, so 1080p only added USB/decode/resize cost.
All downstream consumers use NORMALIZED coordinates, which are unchanged as
long as the camera's field of view is the same at both resolutions — verify
once on the Pi (arc ny values should match the calibration); if the FOV
differs, pass frame_width=1920, frame_height=1080 here instead of recalibrating.
"""

import math
import threading
import time

from src.hardware.sensors.interfaces import SensorInterface
from src.perception.detector import (
    AluminiumCanDetector, DetectionResult, RUNTIME_MODEL_PATH,
)
from src.perception.target_lock import TargetLock

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
        conf_threshold: float = 0.75,   # keep = AluminiumCanDetector's default
        frame_width: int = 1280,
        frame_height: int = 720,
        device: str = "cpu",
    ):
        # device="cpu" explicitly: auto-select picks CUDA when torch reports it
        # available, and the CUDA build on this Pi imports fine but SIGILLs on
        # the first real op. The live demos have always forced cpu.
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
        self._running.set()          # never leave the worker parked on the wait
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._detector.stop()

    def pause(self):
        """Stop inferring until resume(). The getters keep serving the last
        result, which STALE_AFTER_S retires on its own if the pause runs long."""
        self._running.clear()

    def resume(self):
        self._running.set()

    def _run(self):
        """Capture + infer as fast as the model allows; publish the result."""
        while self._alive:
            # Timed wait so stop() during a pause still ends the thread.
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
        """Cheap by design — inference happens in the worker thread. Kept so
        the polled-sensor contract (update every tick) stays uniform."""
        pass
        
    def _current_target(self):
        """(box, result) of the LOCKED tin — the largest-bbox one, held until
        it leaves the frame. Selection runs once per new inference result, so
        the three getters in one tick all describe the same tin."""
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
        """Forget the current tin now (e.g. after a collection) so the next
        frame re-picks the largest. Normally unnecessary — a collected tin
        leaves the frame and the lock times out on its own."""
        self._target_lock.release()
        self._selected_box = None
        self._selected_at = -1.0

    def get_litter_position(self): # type: ignore
        """
        Returns normalized (x, y) of the LOCKED aluminium can (largest bbox
        area at lock time), where (0.5, 0.5) is the center of the frame.
        Returns None if no can is locked (or detection has gone stale).

        Used by:
            layer1_scan.py    — to check if a target exists
            layer2_approach.py — to calculate motion vector towards the can
        """
        box, result = self._current_target()
        if box is None or result.frame_width == 0:
            return None
        return (box.center_x / result.frame_width,
                box.center_y / result.frame_height)

    def get_litter_ground_contact(self): # type: ignore
        """
        Returns normalized (x, y) of the locked tin's ground-contact point
        (bbox bottom-center), or None. Arc-grasp and pixel_to_arm
        calibrations are anchored to this point.

        Used by:
            layer3_collect.py — to solve the grasp pose
        """
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

    def get_litter_pose(self): # type: ignore
        """
        Returns the locked tin's pose estimated from its segmentation mask:
        {'klass': 'upright'|'lying'|'axial', 'angle': deg 0..180},
        or None when there is no target / no usable mask.

        Used by:
            layer3_collect.py — upright vs lying picks the grasp grid, and
            the angle drives the wrist roll (CH5) for lying tins.
        """
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
        """Returns the newest DetectionResult (even if stale) for debugging."""
        with self._result_lock:
            return self._latest_result

    def get_annotated_frame(self):
        """
        Returns the last camera frame with bounding boxes drawn, the locked
        target highlighted. Use with cv2.imshow() or the dashboard stream.
        """
        return self._detector.get_annotated_frame(self.get_latest_result(), target=self._target_lock.target)

    def read_camera_frame(self):
        """
        Opens the camera (if not already open) and returns one raw frame —
        no model load, no inference. For sanity-checking the camera feed by
        itself, e.g. cv2.imshow() from a manual test script.
        """
        if not self._detector._cap or not self._detector._cap.isOpened():
            self._detector._open_camera()
        return self._detector.read_frame()
