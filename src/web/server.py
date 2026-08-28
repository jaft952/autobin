"""
web/server.py

AutoBin robot server — runs ON THE PI (the hardware is there), exposing the
control/telemetry API around RobotRuntime. The React dashboard is hosted
separately on Windows (web/dashboard.py) and connects here directly.

Run on the Pi (needs: pip install flask):

    python web/server.py                     # full robot, port 8000
    python web/server.py --no-camera         # bench without YOLO/webcam

Then on Windows:   python web/dashboard.py --pi <pi-ip>:8000
(or open http://<pi-ip>:8000 directly — the Pi still serves the UI too,
both deployment modes work).

LOW LATENCY: the dashboard does NOT poll. /api/events is a Server-Sent
Events stream — status is pushed at ~5 Hz and log lines the moment they
appear (SSE = plain HTTP, native in browsers, zero extra pip deps, and CORS
below lets the Windows-hosted UI subscribe cross-origin). Control buttons
are single POSTs (~1-3 ms RTT on LAN). The camera is a continuous MJPEG
stream, unchanged.

API summary (all JSON unless noted):
    GET  /api/status                  state + power + performance snapshot
    POST /api/system/start            full autonomy (AUTO)
    POST /api/system/scan             manual scanning (zigzag only)
    POST /api/system/estop            EMERGENCY STOP
    POST /api/system/shutdown         stop the server and power off the Pi
    POST /api/system/restart          re-exec the server (picks up edited source)
    GET  /api/camera/stream           MJPEG stream (multipart)
    GET  /api/arm/pose                commanded arm pose
    POST /api/arm/pose {name}         home | bin
    POST /api/arm/gripper {action}    open | close
    POST /api/arm/jog {channel,delta} nudge one channel (0-5), degrees
    GET  /api/settings                quick-settings list
    POST /api/settings {key,value}    live-patch one setting (session only)
    GET  /api/logs?after=N            log entries with seq > N
    POST /api/logs/clear
"""
import argparse
import json
import os
import sys
import threading
import time

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_SRC_DIR)                       # for "web.xxx" imports
sys.path.append(os.path.dirname(_SRC_DIR))      # for "src.xxx" imports

from flask import Flask, Response, jsonify, request, send_from_directory # type: ignore

from web.logbuffer import LogBuffer, setup_logging
from web.runtime import RobotRuntime

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(__name__, static_folder=None)
buffer = LogBuffer(capacity=1000)
runtime: RobotRuntime = None          # type: ignore # created in main()


# ── CORS: the dashboard is served from Windows (different origin), so every
#    response — including Flask's automatic OPTIONS preflights — must carry
#    these headers. LAN-only tool, hence the permissive '*'. ───────────────

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def _ok(**extra):
    return jsonify({"ok": True, **extra})


def _fail(exc, code=409):
    return jsonify({"ok": False, "error": str(exc)}), code


# ── Frontend ──────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC_DIR, name)


# ── Low-latency push: SSE stream of status + log entries ─────────────────

STATUS_PUSH_PERIOD = 0.2              # status at 5 Hz
EVENT_POLL_S = 0.05                   # log latency ceiling: ~50 ms


@app.get("/api/events")
def api_events():
    # A reconnect (page refresh, network hiccup, EventSource auto-retry)
    # otherwise always started from 0 and replayed the ENTIRE buffered
    # history again — the client passes back the highest seq it already
    # has so a reconnect only streams what it actually missed.
    start_seq = request.args.get("after", 0, type=int)

    def gen():
        last_seq = start_seq
        next_status = 0.0
        while True:
            now = time.monotonic()

            entries = buffer.since(last_seq)
            if entries:
                last_seq = entries[-1]["seq"]
                yield f"event: logs\ndata: {json.dumps(entries)}\n\n"

            if now >= next_status:
                next_status = now + STATUS_PUSH_PERIOD
                yield f"event: status\ndata: {json.dumps(runtime.status())}\n\n"

            time.sleep(EVENT_POLL_S)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── System control ────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    return jsonify(runtime.status())


@app.post("/api/system/start")
def api_start():
    runtime.start_auto()
    return _ok(state=runtime.state)


@app.post("/api/system/scan")
def api_scan():
    runtime.start_scan()
    return _ok(state=runtime.state)


@app.post("/api/system/estop")
def api_estop():
    runtime.estop()
    return _ok(state=runtime.state)


@app.post("/api/system/shutdown")
def api_shutdown():
    def _later():
        time.sleep(0.5)                     # let the HTTP response flush
        runtime.close()
        os.system("sudo shutdown -h now")

    threading.Thread(target=_later, daemon=True).start()
    return _ok(note="Pi powering off")


@app.post("/api/system/restart")
def api_restart():
    """Re-exec this process so edited source files (layer speeds, thresholds,
    etc.) are picked up on next import — a plain reconnect can't do that,
    the old module stays loaded in memory until the process itself restarts."""
    def _later():
        time.sleep(0.5)                     # let the HTTP response flush
        runtime.close()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_later, daemon=True).start()
    return _ok(note="server restarting")


# ── Camera ────────────────────────────────────────────────────────────────

@app.get("/api/camera/stream")
def api_camera_stream():
    if not runtime.camera_available:
        return jsonify({"ok": False, "error": "no camera"}), 503

    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last = None
        runtime.stream_client_connected()   # runtime only encodes for viewers
        try:
            while True:
                jpeg = runtime.get_frame_jpeg()
                if jpeg is not None and jpeg is not last:
                    yield boundary + jpeg + b"\r\n"
                    last = jpeg
                time.sleep(0.1)             # ~10 fps ceiling
        finally:
            runtime.stream_client_disconnected()

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Manual arm ────────────────────────────────────────────────────────────

@app.get("/api/arm/pose")
def api_arm_pose():
    pose = runtime.arm_pose()
    if pose is None:
        return jsonify({"ok": False, "error": "no arm"}), 503
    return jsonify({"ok": True, **pose})


@app.post("/api/arm/pose")
def api_arm_named_pose():
    try:
        runtime.arm_named_pose((request.get_json(force=True) or {}).get("name", ""))
        return _ok(**runtime.arm_pose()) # type: ignore
    except Exception as exc:
        return _fail(exc)


@app.post("/api/arm/gripper")
def api_arm_gripper():
    try:
        action = (request.get_json(force=True) or {}).get("action", "")
        if action not in ("open", "close"):
            raise ValueError("action must be open|close")
        runtime.arm_gripper(action)
        return _ok(**runtime.arm_pose()) # type: ignore
    except Exception as exc:
        return _fail(exc)


@app.post("/api/arm/jog")
def api_arm_jog():
    try:
        body = request.get_json(force=True) or {}
        runtime.arm_jog(int(body["channel"]), float(body["delta"]))
        return _ok(**runtime.arm_pose()) # type: ignore
    except Exception as exc:
        return _fail(exc)


# ── Quick settings ────────────────────────────────────────────────────────

@app.get("/api/settings")
def api_settings():
    return jsonify({"ok": True, "settings": runtime.get_settings()})


@app.post("/api/settings")
def api_set_setting():
    try:
        body = request.get_json(force=True) or {}
        value = runtime.set_setting(body["key"], float(body["value"]))
        return _ok(key=body["key"], value=value)
    except Exception as exc:
        return _fail(exc, code=400)


# ── Logs ──────────────────────────────────────────────────────────────────

@app.get("/api/logs")
def api_logs():
    after = request.args.get("after", 0, type=int)
    entries = buffer.since(after)
    return jsonify({"ok": True, "entries": entries,
                    "last": entries[-1]["seq"] if entries else after})


@app.post("/api/logs/clear")
def api_logs_clear():
    buffer.clear()
    return _ok()


# ── Entrypoint ────────────────────────────────────────────────────────────

def main():
    global runtime
    ap = argparse.ArgumentParser(description="AutoBin web dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--hz", type=float, default=20.0, help="control loop rate")
    ap.add_argument("--no-camera", action="store_true", help="skip YOLO/webcam")
    args = ap.parse_args()

    log = setup_logging(buffer)
    log.info("AutoBin dashboard starting...")

    runtime = RobotRuntime(log, with_camera=not args.no_camera, hz=args.hz)
    runtime.start()
    log.info(f"dashboard up at http://{args.host}:{args.port} — state STOPPED")

    try:
        # threaded=True: status/log polls must not block the MJPEG stream.
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
