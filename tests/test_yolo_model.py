"""
tests/test_yolo_model.py

Test the aluminium-can YOLO model. Loads the .pt model directly — no ONNX,
no copying anything.

Two modes (auto-detected, or force with --mode):
    windows : CAP_DSHOW backend, 1280x720, uses NVIDIA GPU if present, live window.
    pi      : CAP_V4L2 backend, 640x480 + smaller imgsz for speed, CPU only.
              Falls back to headless (no window) when no display is available.

Usage:
    # Windows (from the project root)
    python tests/test_yolo_model.py                    # webcam 0, live window
    python tests/test_yolo_model.py --source 1         # webcam index 1
    python tests/test_yolo_model.py --source can.jpg   # run on a single image
    python tests/test_yolo_model.py --save out.jpg     # also save the annotated result

    # Raspberry Pi 4 (forces V4L2 + CPU; add --no-show if running over SSH)
    python tests/test_yolo_model.py --mode pi
    python tests/test_yolo_model.py --mode pi --no-show --save out.jpg
"""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "ai/training/scripts/runs/detect/yolov8n_aluminum_can2/weights/best.pt"

# Per-mode capture / inference settings.
MODE_SETTINGS = {
    "windows": {
        "backend": cv2.CAP_DSHOW,   # stable webcam backend on Windows
        "width": 1280,
        "height": 720,
        "imgsz": 640,
    },
    "pi": {
        "backend": cv2.CAP_V4L2,    # standard V4L2 backend on Linux / Pi
        "width": 640,
        "height": 480,
        "imgsz": 320,               # smaller input -> faster on the Pi 4 CPU
    },
}


def detect_mode():
    """windows on Win32, otherwise assume pi (Linux / Raspberry Pi)."""
    return "windows" if sys.platform == "win32" else "pi"


def pick_device(mode):
    """Use the NVIDIA GPU if available on Windows; the Pi 4 has no CUDA -> CPU."""
    if mode == "windows":
        try:
            import torch
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
    return "cpu"


def has_display(mode):
    """Is there a GUI we can draw a window on?"""
    if mode == "windows":
        return True
    # Linux / Pi: need an X / Wayland display, otherwise imshow crashes.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_image(model, path, conf, device, imgsz, show, save, classes):
    img = cv2.imread(str(path))
    if img is None:
        print(f"❌ Could not read image: {path}")
        return
    results = model.predict(source=img, conf=conf, device=device, imgsz=imgsz,
                            classes=classes, verbose=False)
    boxes = results[0].boxes
    print(f"✓ {len(boxes)} detection(s) in {path}")
    for b in boxes:
        name = model.names[int(b.cls[0])]
        print(f"   {name}  conf={float(b.conf[0]):.2f}  xyxy={[int(v) for v in b.xyxy[0]]}")

    annotated = results[0].plot()
    if save:
        cv2.imwrite(save, annotated)
        print(f"✓ annotated image saved to {save}")
    if show:
        cv2.imshow("YOLO test", annotated)
        print("Press any key (in the window) to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(model, index, conf, device, imgsz, settings, show, save, classes):
    cap = cv2.VideoCapture(index, settings["backend"])
    if not cap.isOpened():
        print(f"❌ Cannot open camera index {index}. Try --source 0 or 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings["height"])
    if show:
        print("[Test Started] Press 'q' in the window to quit.")
    else:
        print("[Test Started] Headless mode — press Ctrl+C to quit.")

    last = time.time()
    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Failed to read frame from camera")
                break

            results = model.predict(source=frame, conf=conf, device=device, imgsz=imgsz,
                                     classes=classes, verbose=False)
            annotated = results[0].plot()

            now = time.time()
            fps = 1.0 / (now - last) if now > last else 0.0
            last = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            boxes = results[0].boxes
            if len(boxes):
                best = max(boxes, key=lambda b: float(b.conf[0]))
                name = model.names[int(best.cls[0])]
                cx = int((best.xyxy[0][0] + best.xyxy[0][2]) / 2)
                cy = int((best.xyxy[0][1] + best.xyxy[0][3]) / 2)
                print(f"frame {frame_i}: {len(boxes)} det(s), best={name} {float(best.conf[0]):.2f} "
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
    ap = argparse.ArgumentParser(description="Test the aluminium-can YOLO model (Windows or Pi 4).")
    ap.add_argument("--mode", choices=["auto", "windows", "pi"], default="auto",
                    help="auto-detect by OS, or force 'windows' / 'pi'")
    ap.add_argument("--source", default="0",
                    help="camera index (e.g. 0) or path to an image/video file")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="path to the .pt model")
    ap.add_argument("--conf", type=float, default=0.8, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="inference image size (overrides the per-mode default)")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="only detect these class ids, e.g. --classes 0 (aluminum_can only)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--save", default=None, help="save annotated output to this path")
    args = ap.parse_args()

    mode = detect_mode() if args.mode == "auto" else args.mode
    settings = MODE_SETTINGS[mode]
    imgsz = args.imgsz if args.imgsz is not None else settings["imgsz"]

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    device = pick_device(mode)
    print(f"Mode          : {mode}")
    print(f"Loading model : {model_path}")
    print(f"Device        : {device}")
    print(f"Image size    : {imgsz}")
    model = YOLO(str(model_path))
    # This model was trained with class 0 labeled "item"; show it as "tin" instead.
    # (Cosmetic only — the class id is unchanged, so detection/grasping is unaffected.)
    model.names = {i: ("tin" if i == 0 else n) for i, n in model.names.items()}
    print(f"Classes       : {model.names}")   # the labels baked into THIS model
    print(f"Conf threshold: {args.conf}")
    print("✓ Model loaded")

    # Don't try to open a window if we asked not to, or there's no display (headless Pi / SSH).
    show = not args.no_show and has_display(mode)
    if not args.no_show and not has_display(mode):
        print("⚠ No display detected — running headless (use --save to keep frames).")

    src = args.source
    if src.isdigit():
        run_camera(model, int(src), args.conf, device, imgsz, settings, show, args.save, args.classes)
    else:
        run_image(model, src, args.conf, device, imgsz, show, args.save, args.classes)


if __name__ == "__main__":
    main()
