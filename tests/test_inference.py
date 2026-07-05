"""
tests/test_inference.py — Live YOLO segmentation test with truly decoupled threads.

Three-stage pipeline so the video stays fluid even when inference is slow:

    capture thread    -> always holds the NEWEST camera frame (~30 FPS)
    inference thread  -> runs YOLO on the newest frame, publishes the latest result
    main thread       -> shows the newest frame + last known masks at camera rate

Display FPS and inference FPS are therefore independent numbers: on a
Raspberry Pi the window runs at camera speed while YOLO updates the masks at
whatever rate the CPU manages (~2-6 FPS at imgsz 320). The masks can trail a
fast-moving object by one inference interval — that's the price of drawing
last-known results onto a fresher frame.

Run:
    python tests/test_inference.py                 # defaults: camera 0, imgsz 320
    python tests/test_inference.py --imgsz 640     # slower, more accurate
    python tests/test_inference.py --camera 1

Press 'q' in the video window to quit (or Ctrl+C in the terminal).
Headless SSH note: cv2.imshow needs a desktop session — for headless viewing
use tests/webcam_seg_stream.py, which serves the same feed over HTTP instead.
"""

import argparse
import os
import sys
import threading
import time

import cv2
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.detector import open_camera_capture  # noqa: E402
from src.utils.root import find_repo_root                # noqa: E402

_ROOT = find_repo_root(start_path=__file__)
_DEFAULT_MODEL = _ROOT / "src" / "models" / "yolov11n-seg.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threaded live YOLO segmentation test")
    p.add_argument("--model", default=str(_DEFAULT_MODEL), help="Path to the .pt model")
    p.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    p.add_argument("--imgsz", type=int, default=320,
                   help="YOLO inference size (default: 320 — ~4x faster than 640 on a Pi CPU)")
    p.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    p.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    print("Model loaded.")

    # open_camera_capture picks the right backend (V4L2 on the Pi, DirectShow on
    # Windows), forces MJPG for 30 FPS, and sets BUFFERSIZE=1 so reads are fresh.
    cap, w, h, cam_fps = open_camera_capture(args.camera, args.width, args.height)
    print(f"Camera ready — {w}x{h} @ {cam_fps:.0f}FPS")

    # Shared state between the three threads. One lock guards it all; every
    # critical section is just a reference swap, so contention is negligible.
    lock = threading.Lock()
    state = {"frame": None, "result": None, "infer_fps": 0.0}
    running = threading.Event()
    running.set()

    # ── stage 1: capture — keep only the newest frame ─────────────────────────
    def capture_loop():
        while running.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # mirror view
            with lock:
                state["frame"] = frame

    # ── stage 2: inference — run YOLO on the newest frame, at its own pace ────
    def inference_loop():
        prev = time.time()
        while running.is_set():
            with lock:
                frame = state["frame"]
            if frame is None:
                time.sleep(0.01)
                continue
            results = model.predict(source=frame, conf=args.conf,
                                    imgsz=args.imgsz, verbose=False)
            now = time.time()
            with lock:
                state["result"] = results[0]
                state["infer_fps"] = 1.0 / (now - prev) if now > prev else 0.0
            prev = now

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()

    print("Waiting for first frame …")
    while True:
        with lock:
            if state["frame"] is not None:
                break
        time.sleep(0.05)

    print("\nSystem active — press 'q' in the video window to quit.")

    # ── stage 3: display — full camera rate, last known masks overlaid ────────
    disp_fps = 0.0
    disp_prev = time.time()
    try:
        while True:
            with lock:
                frame = state["frame"]
                result = state["result"]
                infer_fps = state["infer_fps"]

            # Draw the LAST KNOWN detections onto the NEWEST frame. plot(img=...)
            # renders the stored masks/boxes on the array we pass in, so the video
            # stays fluid while the overlay refreshes at inference speed.
            if result is not None:
                annotated = result.plot(img=frame.copy())
            else:
                annotated = frame

            now = time.time()
            inst = 1.0 / (now - disp_prev) if now > disp_prev else 0.0
            disp_prev = now
            disp_fps = 0.9 * disp_fps + 0.1 * inst  # smooth the readout

            cv2.putText(annotated,
                        f"Display: {disp_fps:.0f} FPS   YOLO: {infer_fps:.1f} FPS   imgsz: {args.imgsz}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            try:
                cv2.imshow("YOLO segmentation — threaded (q to quit)", annotated)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print("[!] No display available (headless SSH?). "
                      "Use tests/webcam_seg_stream.py to view over HTTP instead.")
                break

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        running.clear()
        time.sleep(0.2)  # let the worker threads finish their current iteration
        cap.release()
        cv2.destroyAllWindows()
        print("Session ended cleanly.")


if __name__ == "__main__":
    main()
