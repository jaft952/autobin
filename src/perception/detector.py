"""
src/perception/detector.py

Aluminium Can Detector — Perception Layer
Wraps YOLOv8 inference with Logitech C270 webcam.
Returns structured DetectionResult to the sensor interface.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# NOTE: cv2 and ultralytics (which pulls in torch + numpy) are imported lazily
# inside the methods that need them — see _load_model / _open_camera /
# get_annotated_frame. This keeps the pure-Python data classes below importable
# WITHOUT loading any native library, so the offline logic tests run anywhere and
# a broken cv2/torch/numpy wheel (e.g. the Raspberry Pi "Illegal instruction"
# SIGILL) can't crash code that only uses BoundingBox / DetectionResult.


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Pixel-space bounding box of a single detected aluminium can."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center_x(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def center(self) -> tuple:
        """Returns (cx, cy) pixel coordinates."""
        return (self.center_x, self.center_y)

    @property
    def base_center(self) -> tuple:
        """(cx, y2): horizontal center of the bottom edge — where the tin meets
        the floor. This is the tin's ground position, used by IBVS centering as
        the tracked point instead of the bbox center. It is robust to tin height
        and to a tall tin's top leaving the frame in the low, forward-looking
        arm-base camera view."""
        return (self.center_x, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class DetectionResult:
    """
    Output of one inference pass.
    Passed to SensorInterface so Subsumption layers can query detections.
    """
    detections: List[BoundingBox] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0

    @property
    def found(self) -> bool:
        """True if at least one aluminium can is detected."""
        return len(self.detections) > 0

    @property
    def best(self) -> Optional[BoundingBox]:
        """Returns the highest-confidence detection, or None."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)

    def normalized_center(self) -> Optional[tuple]:
        """
        Returns the best detection's center normalized to (0.0–1.0, 0.0–1.0).
        Used by layer2_approach.py for motion vector calculation.
        """
        if not self.best or self.frame_width == 0:
            return None
        nx = self.best.center_x / self.frame_width
        ny = self.best.center_y / self.frame_height
        return (nx, ny)


# ── Camera helper ────────────────────────────────────────────────────────────

def open_camera_capture(camera_index: int = 0,
                        frame_width: int = 1280,
                        frame_height: int = 720):
    """
    Open a cv2.VideoCapture with the correct per-OS backend and the C270's
    high-FPS MJPG setup, then request the given resolution.

    Backend: DirectShow on Windows (needed for a stable C270 connection), V4L2
    on Linux/Raspberry Pi (CAP_DSHOW does not exist there and makes VideoCapture
    fail to open).

    Returns (cap, actual_width, actual_height, actual_fps); the actual values may
    differ from what was requested. Raises RuntimeError if the camera won't open.

    Shared by AluminiumCanDetector and the sweet-spot calibrator so both capture
    with identical geometry — the C270 crops differently per resolution, so
    normalized pixel coords only transfer when the resolution matches.
    """
    import cv2  # lazy: native lib only loaded when a real camera is needed
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {camera_index}. "
            f"Try changing camera_index to 1."
        )
    # MJPG codec — required to reach 30FPS on C270 (default YUYV = 10FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, actual_w, actual_h, actual_fps


# ── Detector ─────────────────────────────────────────────────────────────────

class AluminiumCanDetector:
    """
    YOLOv8-based aluminium can detector using Logitech C270 webcam.

    Usage:
        detector = AluminiumCanDetector(model_path='...', camera_index=0)
        detector.start()

        result = detector.detect()   # call each tick
        if result.found:
            cx, cy = result.best.center

        detector.stop()
    """

    # Promoted "production" copy of the trained weights. NOTE: .pt files are
    # git-ignored, so this 22 MB file must be copied to the Pi manually (it does
    # not arrive via `git pull`). See the README / deploy notes.
    DEFAULT_MODEL_PATH = "ai/models/subsystem2/production/aluminum_can_detector_best.pt"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        camera_index: int = 0,
        conf_threshold: float = 0.5,
        frame_width: int = 1280,
        frame_height: int = 720,
        device=0,
    ):
        self._model_path = Path(model_path)
        self._camera_index = camera_index
        self._conf_threshold = conf_threshold
        self._frame_width = frame_width
        self._frame_height = frame_height
        # 0 = CUDA GPU (training PC). Pass "cpu" on the Raspberry Pi (no CUDA).
        self._device = device

        self._model: Optional[YOLO] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Load model and open camera. Call once before the main loop."""
        self._load_model()
        self._open_camera()
        print("✓ AluminiumCanDetector started")

    def stop(self):
        """Release camera resource. Call on shutdown."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            print("✓ Camera released")

    # ── Core Inference ────────────────────────────────────────────────────

    def detect(self) -> DetectionResult:
        """
        Capture one frame from the camera and run YOLO inference.
        Returns a DetectionResult. Safe to call every tick.
        """
        result = DetectionResult(
            frame_width=self._frame_width,
            frame_height=self._frame_height
        )

        if not self._cap or not self._cap.isOpened():
            print("⚠ Camera not open. Call start() first.")
            return result

        ret, frame = self._cap.read()
        if not ret:
            print("⚠ Failed to read frame from camera")
            return result

        self._last_frame = frame.copy()

        yolo_results = self._model.predict(
            source=frame,
            conf=self._conf_threshold,
            device=self._device,
            verbose=False,
        )

        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                result.detections.append(
                    BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
                )

        return result

    def get_annotated_frame(self, result: DetectionResult):
        """
        Returns the last captured frame with bounding boxes drawn.
        Useful for cv2.imshow() during debugging.
        """
        import cv2  # lazy: only used when drawing debug overlays
        if self._last_frame is None:
            return None

        frame = self._last_frame.copy()

        for det in result.detections:
            # Bounding box
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            # Tracked ground-contact point (bbox bottom-center) — this is the
            # point IBVS centering aligns to the sweet spot.
            bx, by = det.base_center
            cv2.circle(frame, (bx, by), 6, (0, 0, 255), -1)
            # Label
            label = f"Aluminium Can {det.confidence:.2f}"
            cv2.putText(frame, label, (det.x1, det.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Coordinates of the tracked ground point
            cv2.putText(frame, f"({bx},{by})",
                        (bx + 8, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # HUD
        count = len(result.detections)
        cv2.putText(frame, f"Detected: {count} can(s)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return frame

    # ── Private Helpers ───────────────────────────────────────────────────

    def _load_model(self):
        from ultralytics import YOLO  # lazy: only when running live inference
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self._model_path}\n"
                f"Run ai/training/scripts/train_yolov8.py first."
            )
        self._model = YOLO(str(self._model_path))
        print(f"✓ Model loaded: {self._model_path}")

    def _open_camera(self):
        self._cap, actual_w, actual_h, actual_fps = open_camera_capture(
            self._camera_index, self._frame_width, self._frame_height
        )
        # Update actual resolution (may differ from requested)
        self._frame_width = actual_w
        self._frame_height = actual_h
        print(f"✓ Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f}FPS")
