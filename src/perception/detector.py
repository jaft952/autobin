"""
src/perception/detector.py

Aluminium Can Detector — Perception Layer
Wraps YOLOv8 inference with Logitech C270 webcam.
Returns structured DetectionResult to the sensor interface.
"""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.perception.orientation import Orientation, estimate_orientation

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
    orientation: Optional[Orientation] = None   # filled in by detector.infer()

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
    # Keep only the newest frame so slow consumers don't read a stale backlog.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, actual_w, actual_h, actual_fps


def _suppress_contained_boxes(detections, contain_thresh=0.8):
    """Drop any box whose area is >= contain_thresh contained inside a
    higher-confidence box. Removes the 'box-inside-a-box' duplicate detections
    that ordinary NMS keeps when the smaller box's IoU with the bigger one is
    below the NMS threshold (common with a weak / inconsistently-labeled model).
    """
    kept = []
    for d in sorted(detections, key=lambda b: b.confidence, reverse=True):
        d_area = max(1, (d.x2 - d.x1) * (d.y2 - d.y1))
        contained = False
        for k in kept:
            ix1, iy1 = max(d.x1, k.x1), max(d.y1, k.y1)
            ix2, iy2 = min(d.x2, k.x2), min(d.y2, k.y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / d_area >= contain_thresh:
                contained = True
                break
        if not contained:
            kept.append(d)
    return kept


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

    DEFAULT_MODEL_PATH = "src/models/best.pt"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        camera_index: int = 0,
        conf_threshold: float = 0.8,
        frame_width: int = 1280,
        frame_height: int = 720,
        device=0,
        imgsz: int = 640,
    ):
        self._model_path = Path(model_path)
        self._camera_index = camera_index
        self._conf_threshold = conf_threshold
        self._frame_width = frame_width
        self._frame_height = frame_height
        # 0 = CUDA GPU (training PC). Pass "cpu" on the Raspberry Pi (no CUDA).
        self._device = device
        # YOLO inference image size. Smaller (e.g. 320) is much faster on the Pi
        # CPU; detections are still returned in full-frame pixel coords.
        self._imgsz = imgsz

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

    def read_frame(self):
        """
        Grab one frame from the camera WITHOUT running inference (cheap). Updates
        the stored frame used by get_annotated_frame(). Returns the BGR frame, or
        None if the camera isn't open / the read failed. Lets callers show the
        camera at full rate but run YOLO less often.
        """
        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        self._last_frame = frame
        return frame

    def infer(self, frame) -> DetectionResult:
        """Run YOLO on an already-captured frame and return a DetectionResult."""
        result = DetectionResult(
            frame_width=self._frame_width,
            frame_height=self._frame_height,
        )
        yolo_results = self._model.predict(
            source=frame,
            conf=self._conf_threshold,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                result.detections.append(
                    BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
                )
        # Remove duplicate "box-inside-a-box" detections NMS leaves behind.
        result.detections = _suppress_contained_boxes(result.detections)
        # Classical-CV orientation estimate per detection (upright/lying/axial).
        for d in result.detections:
            d.orientation = estimate_orientation(frame, d)
        return result

    def detect(self) -> DetectionResult:
        """
        Capture one frame from the camera and run YOLO inference.
        Returns a DetectionResult. Safe to call every tick.
        Convenience wrapper = read_frame() + infer().
        """
        frame = self.read_frame()
        if frame is None:
            print("⚠ Camera not open or frame read failed. Call start() first.")
            return DetectionResult(
                frame_width=self._frame_width, frame_height=self._frame_height
            )
        return self.infer(frame)

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
            # Label (with box-height % of the frame — helps tune the too_close guard)
            label = f"Can {det.confidence:.2f}  h{det.height / frame.shape[0]:.0%}"
            cv2.putText(frame, label, (det.x1, det.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Coordinates of the tracked ground point
            cv2.putText(frame, f"({bx},{by})",
                        (bx + 8, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Orientation (classical CV): long-axis line + class label (magenta).
            if det.orientation is not None:
                o = det.orientation
                cx, cy = det.center
                half = max(det.width, det.height) // 2
                a = math.radians(o.angle)
                dx, dy = int(half * math.cos(a)), int(half * math.sin(a))
                cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 0, 255), 2)
                cv2.putText(frame, f"{o.klass} {o.angle:.0f}deg",
                            (det.x1, min(det.y2 + 20, frame.shape[0] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

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
