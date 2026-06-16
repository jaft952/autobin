"""
tests/test_yolo_model.py

Test the aluminium-can YOLO model on THIS Windows PC (where torch / ultralytics
work). Loads the .pt model directly — no ONNX, no copying anything.

Usage (Windows, from the project root):
    python tests/test_yolo_model.py                    # webcam 0, live window
    python tests/test_yolo_model.py --source 1         # webcam index 1
    python tests/test_yolo_model.py --source can.jpg   # run on a single image
    python tests/test_yolo_model.py --save out.jpg     # also save the annotated result
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "ai/models/subsystem2/production/aluminum_can_detector_best.pt"


def pick_device():
    """Use the NVIDIA GPU if available, otherwise CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def run_image(model, path, conf, device, show, save, classes):
    img = cv2.imread(str(path))
    if img is None:
        print(f"❌ Could not read image: {path}")
        return
    results = model.predict(source=img, conf=conf, device=device, classes=classes, verbose=False)
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


def run_camera(model, index, conf, device, show, save, classes):
    # CAP_DSHOW is the stable webcam backend on Windows.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ Cannot open camera index {index}. Try --source 0 or 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[Test Started] Press 'q' in the window to quit.")

    last = time.time()
    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Failed to read frame from camera")
                break

            results = model.predict(source=frame, conf=conf, device=device,
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
    ap = argparse.ArgumentParser(description="Test the aluminium-can YOLO model on Windows.")
    ap.add_argument("--source", default="0",
                    help="camera index (e.g. 0) or path to an image/video file")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="path to the .pt model")
    ap.add_argument("--conf", type=float, default=0.8, help="confidence threshold")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="only detect these class ids, e.g. --classes 0 (aluminum_can only)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--save", default=None, help="save annotated output to this path")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    device = pick_device()
    print(f"Loading model : {model_path}")
    print(f"Device        : {device}")
    model = YOLO(str(model_path))
    print(f"Classes       : {model.names}")   # the labels baked into THIS model
    print(f"Conf threshold: {args.conf}")
    print("✓ Model loaded")

    show = not args.no_show
    src = args.source
    if src.isdigit():
        run_camera(model, int(src), args.conf, device, show, args.save, args.classes)
    else:
        run_image(model, src, args.conf, device, show, args.save, args.classes)


if __name__ == "__main__":
    main()
