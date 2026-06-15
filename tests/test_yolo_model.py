"""
tests/test_yolo_model.py

Test the aluminium-can YOLO model on the Raspberry Pi using ONLY OpenCV
(cv2.dnn) + numpy. It does NOT import torch / torchvision / ultralytics, which
crash with "Illegal instruction" on this Pi.

Prerequisite: an ONNX version of the model. Create it once on the Windows PC:
    python tests/export_to_onnx.py
then copy the resulting .onnx next to the .pt (same folder) on the Pi.

Usage (on the Pi):
    python3 tests/test_yolo_model.py                       # webcam 0, live window if a display exists
    python3 tests/test_yolo_model.py --source 1            # webcam index 1
    python3 tests/test_yolo_model.py --source can.jpg      # run on a single image
    python3 tests/test_yolo_model.py --no-show --save out.jpg   # headless, save annotated frame
"""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "ai/models/subsystem2/production/aluminum_can_detector_best.onnx"

INPUT_SIZE = 640                 # must match imgsz used in export_to_onnx.py
CLASS_NAMES = ["aluminium can"]  # single-class detector; index 0


def has_display() -> bool:
    """cv2.imshow needs an X display on Linux; assume yes on Windows/macOS."""
    if os.environ.get("DISPLAY"):
        return True
    return os.name != "posix"


def letterbox(img, size=INPUT_SIZE, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to a square. Returns (canvas, scale, dx, dy)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    dy, dx = (size - nh) // 2, (size - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


def postprocess(output, r, dx, dy, conf_thres, iou_thres):
    """Decode a YOLOv8 ONNX output [1, 4+nc, N] into [(x1,y1,x2,y2,conf,cls), ...]."""
    out = np.squeeze(output)
    if out.ndim != 2:
        return []
    if out.shape[0] < out.shape[1]:   # [4+nc, N] -> [N, 4+nc]
        out = out.T

    cls_scores = out[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    keep = confs >= conf_thres
    out, confs, class_ids = out[keep], confs[keep], class_ids[keep]
    if out.shape[0] == 0:
        return []

    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    x1 = (cx - w / 2 - dx) / r
    y1 = (cy - h / 2 - dy) / r
    boxes = np.stack([x1, y1, w / r, h / r], axis=1).astype(int).tolist()  # x,y,w,h for NMS

    idxs = cv2.dnn.NMSBoxes(boxes, confs.tolist(), conf_thres, iou_thres)
    results = []
    for i in np.array(idxs).flatten():
        x, y, bw, bh = boxes[i]
        results.append((x, y, x + bw, y + bh, float(confs[i]), int(class_ids[i])))
    return results


def infer(net, frame, conf_thres, iou_thres):
    canvas, r, dx, dy = letterbox(frame)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0,
                                 size=(INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    return postprocess(output, r, dx, dy, conf_thres, iou_thres)


def draw(frame, dets):
    for x1, y1, x2, y2, conf, cid in dets:
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def run_image(net, path, conf, iou, show, save):
    img = cv2.imread(str(path))
    if img is None:
        print(f"❌ Could not read image: {path}")
        return
    dets = infer(net, img, conf, iou)
    print(f"✓ {len(dets)} can(s) detected in {path}")
    for x1, y1, x2, y2, c, _ in dets:
        print(f"   conf={c:.2f}  xyxy=({x1},{y1},{x2},{y2})")
    annotated = draw(img, dets)
    if save:
        cv2.imwrite(save, annotated)
        print(f"✓ annotated image saved to {save}")
    if show:
        cv2.imshow("YOLO test", annotated)
        print("Press any key (in the window) to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(net, index, conf, iou, show, save):
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

            dets = infer(net, frame, conf, iou)
            now = time.time()
            fps = 1.0 / (now - last) if now > last else 0.0
            last = now

            annotated = draw(frame, dets)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if dets:
                best = max(dets, key=lambda d: d[4])
                cx, cy = (best[0] + best[2]) // 2, (best[1] + best[3]) // 2
                print(f"frame {frame_i}: {len(dets)} can(s), best conf={best[4]:.2f} "
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
    ap = argparse.ArgumentParser(description="Test the aluminium-can YOLO model with OpenCV only.")
    ap.add_argument("--source", default="0",
                    help="camera index (e.g. 0) or path to an image/video file")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="path to the .onnx model")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--no-show", action="store_true", help="headless: do not open a window")
    ap.add_argument("--save", default=None, help="save annotated output to this path")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ ONNX model not found: {model_path}")
        print("   Create it on the Windows PC with:  python tests/export_to_onnx.py")
        print("   then copy the .onnx to that path on the Pi.")
        sys.exit(1)

    show = (not args.no_show) and has_display()
    if not show and not args.save:
        print("ℹ No display detected; running headless. Use --save out.jpg to keep results.")

    print(f"Loading ONNX model: {model_path}")
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("✓ Model loaded (cv2.dnn, CPU)")

    src = args.source
    if src.isdigit():
        run_camera(net, int(src), args.conf, args.iou, show, args.save)
    else:
        run_image(net, src, args.conf, args.iou, show, args.save)


if __name__ == "__main__":
    main()
