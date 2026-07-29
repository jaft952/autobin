"""
Aluminium Can Detector — YOLO11n-seg with Logitech Brio 4K.
Lazy imports keep pure-Python data classes importable without torch/cv2.
"""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
from ultralytics import YOLO

from src.perception.orientation import Orientation, estimate_orientation

# THE production model — single source of truth. CameraSensor (runtime),
# the calibration tools and the NCNN export all read this constant, so
# switching to a newly trained checkpoint is a ONE-LINE change here.
RUNTIME_MODEL_PATH = "src/models/yolov11n-seg.pt"


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    orientation: Optional[Orientation] = None
    mask_poly: Optional[object] = None  # YOLO11n-seg segmentation outline (Nx2 pixel coords)

    @property
    def center_x(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def center(self) -> tuple:
        return (self.center_x, self.center_y)

    @property
    def base_center(self) -> tuple:
        return (self.center_x, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def mask_area(self) -> Optional[int]:
        """Actual pixel count of segmented can."""
        if self.mask_poly is None:
            return None
        import numpy as np
        import cv2
        poly = np.asarray(self.mask_poly, dtype=np.int32).reshape(-1, 2)
        return int(cv2.contourArea(poly))


@dataclass
class DetectionResult:
    detections: List[BoundingBox] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0

    @property
    def found(self) -> bool:
        return len(self.detections) > 0

    @property
    def best(self) -> Optional[BoundingBox]:
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)

    def normalized_center(self) -> Optional[tuple]:
        if not self.best or self.frame_width == 0:
            return None
        return (
            self.best.center_x / self.frame_width,
            self.best.center_y / self.frame_height,
        )

    def normalized_base_center(self) -> Optional[tuple]:
        """Ground-contact point (bbox bottom-center), normalized. This is
        what the arc-grasp calibration and pixel_to_arm homography are
        anchored to — the tin touches the floor here, not at the bbox
        center."""
        if not self.best or self.frame_width == 0:
            return None
        u, v = self.best.base_center
        return (u / self.frame_width, v / self.frame_height)


def open_camera_capture(camera_index: int = 0,
                        frame_width: int = 1920,
                        frame_height: int = 1080):
    """Open Logitech Brio 4K with platform-specific backend."""
    import cv2
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    # After a mid-run USB drop the camera often re-enumerates at a new index,
    # so fall back through a few before giving up.
    tried = []
    cap = None
    for idx in dict.fromkeys([camera_index, 0, 1, 2]):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            if idx != camera_index:
                print(f"[camera] index {camera_index} unavailable — using index {idx}")
            break
        cap.release()
        cap = None
        tried.append(idx)
    if cap is None:
        raise RuntimeError(
            f"Cannot open camera (tried indices {tried}). If it worked before, the "
            f"camera likely dropped off USB (power sag) — replug it or run: "
            f"sudo modprobe -r uvcvideo && sudo modprobe uvcvideo")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, actual_w, actual_h, actual_fps


def _filter_contained_boxes(detections, contain_thresh=0.8):
    """Remove boxes that are ≥80% contained inside higher-confidence boxes."""
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


class AluminiumCanDetector:
    DEFAULT_MODEL_PATH = RUNTIME_MODEL_PATH

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        camera_index: int = 0,
        conf_threshold: float = 0.60,
        frame_width: int = 1280,
        frame_height: int = 720,
        device=None,
        imgsz: int = 640,
        use_ncnn: bool = True,
    ):
        self._model_path = Path(model_path)
        self._camera_index = camera_index
        self._conf_threshold = conf_threshold
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._device = device  # None = auto (CUDA if available, else CPU)
        self._imgsz = imgsz
        self._use_ncnn = use_ncnn  # False forces .pt even if an NCNN export exists

        self._model: Optional[YOLO] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame = None

    def start(self):
        """Load model and open camera."""
        self._load_model()
        self._open_camera()
        print("✓ AluminiumCanDetector started")

    def stop(self):
        """Release camera resource."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            print("✓ Camera released")

    def read_frame(self):
        """Grab frame without inference; updates cache for get_annotated_frame()."""
        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if ret:
            self._last_frame = frame
        return frame if ret else None

    def infer(self, frame, imgsz: Optional[int] = None,
              fast: bool = False) -> DetectionResult:
        """Run YOLO11n-seg and return DetectionResult with segmentation masks.

        imgsz: per-call inference size override (e.g. 320 for a faster, less
            accurate pass). NCNN exports may reject a size other than the one
            they were exported at — on failure this falls back to the default
            size permanently (logged once).
        fast: skip per-detection orientation estimation (upright/lying) —
            saves time when the caller only tracks position (cruise approach).
        """
        result = DetectionResult(frame_width=self._frame_width, frame_height=self._frame_height)
        size = imgsz or self._imgsz
        if imgsz and getattr(self, "_imgsz_override_broken", False):
            size = self._imgsz
        try:
            yolo_results = self._model.predict( # type: ignore
                source=frame,
                conf=self._conf_threshold,
                device=self._resolve_device(),
                imgsz=size,
                verbose=False,
            )
        except Exception as exc:
            if size == self._imgsz:
                raise           # the normal size failed — a real error
            # The override size was rejected (NCNN exports are fixed-shape):
            # remember and permanently fall back to the export size.
            self._imgsz_override_broken = True
            print(f"[detector] imgsz={size} rejected by this model backend "
                  f"({exc}); staying at {self._imgsz}.")
            return self.infer(frame, imgsz=None, fast=fast)
        for r in yolo_results:
            polys = r.masks.xy if r.masks is not None else None
            for i, box in enumerate(r.boxes): # type: ignore
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                poly = polys[i] if polys and i < len(polys) and len(polys[i]) >= 3 else None
                result.detections.append(
                    BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, mask_poly=poly)
                )
        result.detections = _filter_contained_boxes(result.detections)
        if not fast:
            for d in result.detections:
                d.orientation = estimate_orientation(frame, d)
        return result

    def detect(self) -> DetectionResult:
        """Capture and infer in one call."""
        frame = self.read_frame()
        if frame is None:
            return DetectionResult(frame_width=self._frame_width, frame_height=self._frame_height)
        return self.infer(frame)

    def get_annotated_frame(self, result: DetectionResult):
        """Render detections with segmentation masks, bbox, base_center, and orientation."""
        import cv2
        import numpy as np
        if self._last_frame is None:
            return None

        frame = self._last_frame.copy()
        for det in result.detections:
            if det.mask_poly is not None:
                pts = np.asarray(det.mask_poly, dtype=np.int32).reshape(-1, 2)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 200, 0))
                cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, dst=frame)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)

            bx, by = det.base_center
            cv2.circle(frame, (bx, by), 6, (0, 0, 255), -1)

            label = f"Can {det.confidence:.2f}  h{det.height / frame.shape[0]:.0%}"
            cv2.putText(frame, label, (det.x1, det.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"({bx},{by})", (bx + 8, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

        count = len(result.detections)
        cv2.putText(frame, f"Detected: {count} can(s)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return frame

    def _resolve_device(self):
        """None -> CUDA if this machine has it, else CPU."""
        if self._device is not None:
            return self._device
        try:
            import torch
            self._device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            self._device = "cpu"
        print(f"✓ Inference device auto-selected: {self._device}")
        return self._device

    def _pick_model_path(self) -> Path:
        """Prefer an NCNN export next to the .pt (2-4x faster on the Pi's ARM CPU)."""
        if self._use_ncnn and self._model_path.suffix == ".pt":
            ncnn = self._model_path.with_name(self._model_path.stem + "_ncnn_model")
            if ncnn.is_dir():
                print(f"✓ Using NCNN export: {ncnn}")
                return ncnn
        if not self._use_ncnn:
            print(f"✓ NCNN disabled (use_ncnn=False) — using {self._model_path}")
        return self._model_path

    def _load_model(self):
        """Load YOLO11n-seg checkpoint (NCNN export preferred) and warm up."""
        from ultralytics import YOLO
        import numpy as np
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")
        path = self._pick_model_path()
        # task="segment": the NCNN export has no embedded task metadata and
        # defaults to "detect", which misparses this model's seg output.
        self._model = YOLO(str(path), task="segment")
        print(f"✓ Model loaded: {path}")
        # Warmup: the first predict pays one-off graph/init cost (hundreds of
        # ms); do it here on a dummy frame so the first real tick is fast.
        self._model.predict(
            source=np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8),
            device=self._resolve_device(), imgsz=self._imgsz, verbose=False,
        )

    def _open_camera(self):
        """Open Brio 4K camera with actual resolution."""
        self._cap, actual_w, actual_h, actual_fps = open_camera_capture(
            self._camera_index, self._frame_width, self._frame_height
        )
        self._frame_width = actual_w
        self._frame_height = actual_h
        print(f"✓ Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f}FPS")
