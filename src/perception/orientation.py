"""Works out if a can is upright or lying."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Orientation:
    angle: float
    aspect: float
    klass: str


def _segment_bbox_crop(frame_bgr, box):
    """Find the can's outline inside its box."""
    import cv2
    import numpy as np

    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = int(box.x2), int(box.y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) < 0.15 * mask.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(mask) > 0.5 * mask.size:
            mask = cv2.bitwise_not(mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.05 * mask.size:
        return None
    return cnt


def estimate_orientation(frame_bgr, box, round_aspect: float = 1.35,
                         upright_band_deg: float = 35.0):
    """Estimate the can's orientation."""
    import cv2
    import numpy as np

    poly = getattr(box, "mask_poly", None)
    if poly is not None and len(poly) >= 3:
        cnt = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    else:
        cnt = _segment_bbox_crop(frame_bgr, box)
        if cnt is None:
            return None

    (_, (w, h), _) = cv2.minAreaRect(cnt)
    if w == 0 or h == 0:
        return None
    pts = cv2.boxPoints(cv2.minAreaRect(cnt))

    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    p1, p2 = max(edges, key=lambda e: float(np.hypot(*(e[1] - e[0]))))
    angle = float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0)
    aspect = float(max(w, h) / min(w, h))

    if aspect < round_aspect:
        klass = "axial"
    elif abs(angle - 90.0) <= upright_band_deg:
        klass = "upright"
    else:
        klass = "lying"

    return Orientation(angle=round(angle, 1), aspect=round(aspect, 2), klass=klass)
