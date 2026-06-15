"""
tests/test_yolo_model.py

Quick test for the aluminium-can YOLO model. Built to run on the Raspberry Pi
(CPU, optionally headless) as well as the dev PC.

This is STANDALONE on purpose: it uses ultralytics + OpenCV directly instead of
src/perception/detector.py, so it does NOT inherit that file's Windows-only
camera backend (cv2.CAP_DSHOW) or its CUDA default (device=0). Device is picked
automatically (CPU on the Pi).

Usage (on the Pi):
    python3 tests/test_yolo_model.py                       # webcam 0, live window if a display exists
    python3 tests/test_yolo_model.py --source 1            # webcam index 1
    python3 tests/test_yolo_model.py --source can.jpg      # run on a single image
    python3 tests/test_yolo_model.py --no-show --save out.jpg   # headless, save annotated frame

If you still get "Illegal instruction" at startup, run tests/diagnose_imports.py
first to find which native library is crashing.
"""

# OpenBLAS on the Pi sometimes auto-selects a CPU kernel that triggers
# "Illegal instruction". This MUST be set before numpy/torch/cv2 are imported,
# so it stays at the very top of the file.
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import sys
import time
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "ai/models/subsystem2/production/aluminum_can_detector_best.pt"


def pick_device():
    """Use the GPU on the training PC, fall back to CPU on the Pi."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def has_display() -> bool:
    """cv2.imshow needs an X display on Linux; assume yes on Windows/macOS."""
    if os.environ.get("DISPLAY"):
        return True
    return os.name != "posix"


def run_image(model, path, conf, device, show, save):
    img = cv2.imread(str(path))
    if img is None:
        print(f"❌ Could not read image: {path}")
        return
    results = model.predict(source=img, conf=conf, device=device, verbose=False)
    boxes = results[0].boxes
    print(f"✓ {len(boxes)} can(s) detected in {path}")
    for b in boxes:
        print(f"   conf={float(b.conf[0]):.2f}  xyxy={[int(v) for v in b.xyxy[0]]}")

    annotated = results[0].plot()
    if save:
        cv2.imwrite(save, annotated)
        print(f"✓ annotated image saved to {save}")
    if show:
        cv2.imshow("YOLO test", annotated)
        print("Press any key (in the window) to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(model, index, conf, device, show, save):
    cap = cv2.VideoCapture(index)  # default backend (V4L2 on the Pi)
    if not cap.isOpened():
        print(f"❌ Cannot open camera index {index}. "
              f"Try --source 0 or 1, or check `ls /dev/video*` on the Pi.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[Test Started] Press 'q' (in the window) or Ctrl+C to quit.")

    last = time.time()
    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Failed to read frame from camera")
                break

            results = model.predict(source=frame, conf=conf, device=device, verbose=False)
            annotated = results[0].plot()

            now = time.time()
            fps = 1.0 / (now - last) if now > last else 0.0
            last = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            n = len(results[0].boxes)
            if n:
                best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                cx = int((best.xyxy[0][0] + best.xyxy[0][2]) / 2)
                cy = int((best.xyxy[0][1] + best.xyxy[0][3]) / 2)
                print(f"frame {frame_i}: {n} can(s), best conf={float(best.conf[0]):.2f} "
                      f"center=({cx},{cy}) fps={fps:.1f}")

            if show:
                cv2.imshow("YOLO test", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif save:
                cv2.imwrite(save, annotated)  # keep overwriting the latest frame
            frame_i += 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Test the aluminium-can YOLO model.")
    ap.add_argument("--source", default="0",
                    help="camera index (e.g. 0) or path to an image/video file")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="path to the .pt model")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    ap.add_argument("--no-show", action="store_true", help="headless: do not open a window")
    ap.add_argument("--save", default=None, help="save annotated output to this path")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    device = pick_device()
    show = (not args.no_show) and has_display()
    if not show and not args.save:
        print("ℹ No display detected; running headless. Use --save out.jpg to keep results.")

    print(f"Loading model : {model_path}")
    print(f"Device        : {device}")
    model = YOLO(str(model_path))
    print("✓ Model loaded")

    src = args.source
    if src.isdigit():
        run_camera(model, int(src), args.conf, device, show, args.save)
    else:
        run_image(model, src, args.conf, device, show, args.save)


if __name__ == "__main__":
    main()
