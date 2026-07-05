"""
tests/webcam_seg_stream.py  —  Live YOLO Segmentation Test (headless / Raspberry Pi)

Runs a YOLO segmentation model against a live webcam feed and serves the
annotated frames as an MJPEG stream over HTTP. Designed for a headless
Raspberry Pi (SSH only, no monitor attached): start this on the Pi, then open
the printed URL in a browser on any device on the same network to watch the
segmentation output live.

Run on the Pi:
    python tests/webcam_seg_stream.py --device cpu

Then browse to the printed http://<pi-ip>:8080/ from your laptop/phone.

Run locally on Windows for a quick sanity check before deploying:
    python tests/webcam_seg_stream.py

Press Ctrl+C in the terminal to stop.
"""

import argparse
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from ultralytics import YOLO

from src.perception.detector import open_camera_capture
from src.utils.root import find_repo_root

_ROOT = find_repo_root(start_path=__file__)
_DEFAULT_MODEL = _ROOT / "src" / "models" / "yolov11n-seg.pt"

# Updated by the inference loop, read by the HTTP handler on every request.
_latest_jpeg: bytes | None = None
_latest_jpeg_lock = threading.Lock()


# ─────────────────────────── logging setup ─────────────────────────────────

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("AutobinSegStream")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(fmt="%(asctime)s  [%(levelname)s]  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ─────────────────────────── CLI args ──────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live YOLO segmentation test, streamed over HTTP as MJPEG")
    p.add_argument("--model", default=str(_DEFAULT_MODEL), help="Path to the .pt segmentation model")
    p.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    p.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (default: 640; use 320 for more FPS on a Pi CPU)")
    p.add_argument("--device", default="cpu", help="Inference device: 'cpu' on the Pi, 0 for a CUDA GPU (default: cpu)")
    p.add_argument("--classes", default=None, help="Comma-separated class IDs to keep, e.g. '39,41' (default: all classes)")
    p.add_argument("--host", default="0.0.0.0", help="HTTP bind address (default: 0.0.0.0 — reachable from other devices)")
    p.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG encode quality for the stream (default: 80)")
    return p.parse_args()


def get_lan_ip() -> str:
    """Best-effort LAN IP so the printed URL is reachable from another device.
    Opens no real connection — UDP connect() just makes the OS pick the outbound
    interface, which is more reliable on a Pi than gethostbyname (often 127.0.1.1)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ─────────────────────────── MJPEG HTTP server ─────────────────────────────

class _StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default per-request access logging; the inference loop already logs

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _latest_jpeg_lock:
                        frame = _latest_jpeg
                    if frame is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.03)  # ~30fps cap on the serving side
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab / navigated away
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='margin:0;background:#111'>"
                b"<img src='/stream' style='width:100%;height:auto;display:block'>"
                b"</body></html>"
            )


def start_stream_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _StreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ─────────────────────────── helpers ───────────────────────────────────────

def mask_pixel_area(polygon_xy) -> float:
    """Pixel area of one segmentation instance from its polygon (frame-space xy)."""
    import numpy as np
    if polygon_xy is None or len(polygon_xy) < 3:
        return 0.0
    return float(cv2.contourArea(np.array(polygon_xy, dtype="float32")))


# ─────────────────────────── main ──────────────────────────────────────────

def main() -> None:
    args = parse_args()
    log = setup_logger()
    class_filter = [int(c) for c in args.classes.split(",")] if args.classes else None

    log.info("=" * 60)
    log.info("Autobin — Live Segmentation Stream")
    log.info(f"  Model      : {args.model}")
    log.info(f"  Confidence : {args.conf}")
    log.info(f"  Image size : {args.imgsz}")
    log.info(f"  Device     : {args.device}")
    log.info(f"  Classes    : {class_filter or 'all'}")
    log.info("=" * 60)

    log.info(f"Loading model from: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as exc:
        log.error(f"Failed to load model: {exc}")
        return
    log.info("Model loaded successfully.")

    log.info(f"Opening camera (index {args.camera}) …")
    try:
        cap, w, h, fps = open_camera_capture(args.camera, args.width, args.height)
    except RuntimeError as exc:
        log.error(str(exc))
        return
    log.info(f"Camera ready — resolution: {w}x{h} @ {fps:.0f}FPS")

    server = start_stream_server(args.host, args.port)
    lan_ip = get_lan_ip()
    log.info(f"Stream ready — open http://{lan_ip}:{args.port}/ on another device on the same network")
    log.info("Press Ctrl+C to stop.\n")

    frame_count = 0
    fps_timer = time.time()
    fps_display = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                log.error("Failed to grab frame — ending stream.")
                break
            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                fps_display = 30 / elapsed if elapsed > 0 else 0
                fps_timer = time.time()

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                classes=class_filter,
                verbose=False,
            )
            r = results[0]
            annotated = r.plot()  # draws masks + boxes for a segmentation model

            instance_count = 0
            if r.masks is not None:
                instance_count = len(r.masks.xy)
                names = model.names
                for i in range(instance_count):
                    cls_id = int(r.boxes.cls[i])
                    conf_val = float(r.boxes.conf[i])
                    area = mask_pixel_area(r.masks.xy[i])
                    log.info(
                        f"SEGMENTED  |  {names.get(cls_id, f'class_{cls_id}'):<20}  "
                        f"conf={conf_val:.2f}  mask_area={area:>8.0f} px²"
                    )

            overlay_lines = [
                f"FPS: {fps_display:.1f}",
                f"Frame: {frame_count}",
                f"Instances: {instance_count}",
            ]
            for i, txt in enumerate(overlay_lines):
                cv2.putText(annotated, txt, (10, 25 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if ok:
                with _latest_jpeg_lock:
                    global _latest_jpeg
                    _latest_jpeg = jpeg.tobytes()

    except KeyboardInterrupt:
        log.info("Stop requested by user.")
    finally:
        cap.release()
        server.shutdown()
        log.info("─" * 60)
        log.info(f"Session ended | frames processed: {frame_count}")
        log.info("─" * 60)


if __name__ == "__main__":
    main()
