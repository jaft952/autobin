"""
detect_log.py  —  YOLOv8 Real-Time Detection with Terminal Logging
Run inside your venv:  python detect_log.py [--model PATH] [--conf 0.88] [--area 8000]
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO
from src.utils.root import find_repo_root

# Repo-root anchor so the default model path works no matter where this runs from.
# This file is at ai/training/scripts/subsystem1/ -> parents[4] is the repo root.
# _ROOT = Path(__file__).resolve().parents[4]
_ROOT = find_repo_root(start_path=__file__)
# _DEFAULT_MODEL = _ROOT / "ai" / "models" / "subsystem1" / "production" / "inference_01072026.pt"
_DEFAULT_MODEL = _ROOT / "src" / "models" / "yolov11n-seg.pt"

# ─────────────────────────── logging setup ────────────────────────────────────

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("AutobinDetect")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%H:%M:%S",
    )

    # console handler (stdout)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # optional file handler — saves a copy next to this script
    # log_path = Path(__file__).parent / "detections.log"
    # fh = logging.FileHandler(log_path, encoding="utf-8")
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(fmt)
    # logger.addHandler(fh)

    return logger


# ─────────────────────────── CLI args ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 live detection with logging")
    p.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        help="Path to the .pt model file",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.8,
        help="Confidence threshold for detection (default: 0.88)",
    )
    p.add_argument(
        "--area",
        type=int,
        default=50000,
        help="Bounding-box area (px²) that triggers a LARGE OBJECT warning (default: 8000)",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Camera device index (default: 0)",
    )
    p.add_argument(
        "--no-window",
        action="store_true",
        help="Run headless — no OpenCV display window",
    )
    return p.parse_args()


# ─────────────────────────── helpers ──────────────────────────────────────────

def bbox_area(box) -> int:
    """Return pixel area of a single ultralytics bounding box."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return int((x2 - x1) * (y2 - y1))


# ─────────────────────────── main ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    log = setup_logger()

    log.info("=" * 60)
    log.info("Autobin  —  YOLOv8 Detection Logger")
    log.info(f"  Model      : {args.model}")
    log.info(f"  Confidence : {args.conf}")
    log.info(f"  Area alert : {args.area} px²")
    log.info("=" * 60)

    # ── load model ────────────────────────────────────────────────────────────
    log.info(f"Loading model from: {args.model}")
    try:
        model = YOLO(args.model)
        log.info("Model loaded successfully.")
    except Exception as exc:
        log.error(f"Failed to load model: {exc}")
        return

    # ── open camera ───────────────────────────────────────────────────────────
    log.info(f"Opening camera (index {args.camera}) …")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        log.error("Could not open webcam. Check camera index or connection.")
        return

    # read one frame to get resolution
    ok, probe = cap.read()
    if ok:
        h, w = probe.shape[:2]
        log.info(f"Camera ready  —  resolution: {w}x{h}")
        frame_area = w * h
        log.info(f"Frame area: {frame_area} px²  |  Alert threshold: {args.area} px²")
    else:
        log.warning("Could not read probe frame; resolution unknown.")

    log.info("Press  q  in the video window to quit.\n")

    # ── detection state ───────────────────────────────────────────────────────
    frame_count   = 0
    detect_count  = 0       # total detections across all frames
    fps_timer     = time.time()
    fps_display   = 0.0

    # Track the previous detection count per frame to log only on change
    prev_frame_detections = -1

    # ── main loop ─────────────────────────────────────────────────────────────
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            log.error("Failed to grab frame — ending stream.")
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # ── FPS calculation (every 30 frames) ─────────────────────────────────
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            fps_display = 30 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        # ── inference ─────────────────────────────────────────────────────────
        results = model.predict(source=frame, conf=args.conf, verbose=False)

        annotated_frame = frame
        frame_detections = 0

        if results and results[0].boxes:
            boxes = results[0].boxes
            names = model.names            # {class_id: class_name}
            frame_detections = len(boxes)
            detect_count += frame_detections
            annotated_frame = results[0].plot()

            # ── per-box logging ────────────────────────────────────────────
            for box in boxes:
                cls_id    = int(box.cls[0])
                cls_name  = names.get(cls_id, f"class_{cls_id}")
                conf_val  = float(box.conf[0])
                area      = bbox_area(box)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                log.info(
                    f"DETECTED  |  {cls_name:<20}  "
                    f"conf={conf_val:.2f}  "
                    f"area={area:>7} px²  "
                    f"bbox=({x1},{y1})-({x2},{y2})"
                )

                if area >= args.area:
                    log.warning(
                        f"LARGE OBJECT  |  {cls_name}  "
                        f"area={area} px²  >=  threshold={args.area} px²"
                    )

        # ── frame summary (only when detection count changes) ─────────────────
        if frame_detections != prev_frame_detections:
            if frame_detections == 0 and prev_frame_detections > 0:
                log.info("--- No objects detected ---")
            prev_frame_detections = frame_detections

        # ── overlay stats on frame ─────────────────────────────────────────────
        if not args.no_window:
            overlay_lines = [
                f"FPS: {fps_display:.1f}",
                f"Frame: {frame_count}",
                f"Detections this frame: {frame_detections}",
                f"Total detections: {detect_count}",
                f"Area threshold: {args.area} px2",
            ]
            for i, txt in enumerate(overlay_lines):
                cv2.putText(
                    annotated_frame, txt,
                    (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA,
                )

            cv2.imshow("YOLOv8 Detection Logger  (press q to quit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Quit requested by user.")
                break

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    log.info("─" * 60)
    log.info(f"Session ended  |  frames processed: {frame_count}  |  total detections: {detect_count}")
    log.info("─" * 60)


if __name__ == "__main__":
    main()
