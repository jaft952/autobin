"""
Verification for IBVS Stage-1 chassis centering (src/visual_servoing/ibvs_centering.py).

Two modes:
  python tests/test_ibvs_centering.py
      OFFLINE logic checks with synthetic detections — no camera, no YOLO model,
      runs anywhere. Exit code 0 = all passed.

  python tests/test_ibvs_centering.py --live
      LIVE mode on the Pi: opens the camera + YOLO (on CPU), prints the centering
      status every frame, and shows the annotated video.

  python tests/test_ibvs_centering.py --sim
      SIM mode on the Pi: real camera feed but NO YOLO. You place a fake "upright
      tin" with the mouse (click or drag) and it STAYS PUT, like a real tin on the
      floor — nothing drifts on its own. The suggested action (FORWARD / BACKWARD /
      LEFT / RIGHT / CATCH ...) is drawn bottom-left, relative to the calibrated
      sweet spot. Lets you verify the centering logic against a live image without
      the slow, flaky detector. q = quit.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.detector import BoundingBox, DetectionResult
from src.visual_servoing.ibvs_centering import (
    ChassisMove,
    CenteringConfig,
    IBVSCentering,
)

W, H = 1280, 720
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def det_at(cx, cy):
    """Synthetic single-tin DetectionResult whose BASE (ground-contact point —
    the bbox bottom-center that IBVSCentering tracks) is at pixel (cx, cy).
    The tin stands ~120px tall upward from that base."""
    box = BoundingBox(x1=cx - 40, y1=cy - 120, x2=cx + 40, y2=cy, confidence=0.9)
    return DetectionResult(detections=[box], frame_width=W, frame_height=H)


def offline():
    cfg = CenteringConfig(ema_alpha=0.0)  # no smoothing -> deterministic asserts
    c = IBVSCentering(cfg)

    print("\n[A] no detection -> SEARCH")
    s = c.update(DetectionResult(frame_width=W, frame_height=H))
    check("move is SEARCH", s.move is ChassisMove.SEARCH, s.message)
    check("not aligned / not stable", not s.aligned and not s.stable)

    print("\n[B] tin dead-center -> HOLD, and stable after debounce")
    c.reset()
    last = None
    for _ in range(cfg.stable_frames):
        last = c.update(det_at(W // 2, H // 2))
    check("aligned", last.aligned)
    check("move is HOLD", last.move is ChassisMove.HOLD)
    check(f"stable after {cfg.stable_frames} frames", last.stable)
    check("motion vector is zero", last.motion_vector == (0.0, 0.0, 0.0))

    print("\n[C] tin near top of image (far ahead) -> FORWARD, negative error_y")
    c.reset()
    s = c.update(det_at(W // 2, int(H * 0.15)))
    check("move is FORWARD", s.move is ChassisMove.FORWARD, s.message)
    check("error_y negative", s.error_y < 0, f"error_y={s.error_y}")
    check("suggests positive forward", s.motion_vector[0] > 0, str(s.motion_vector))

    print("\n[D] tin near bottom (too close) -> BACKWARD")
    c.reset()
    s = c.update(det_at(W // 2, int(H * 0.9)))
    check("move is BACKWARD", s.move is ChassisMove.BACKWARD, s.message)
    check("suggests negative forward", s.motion_vector[0] < 0, str(s.motion_vector))

    print("\n[E] tin to the left -> TURN_LEFT; combined -> FORWARD_LEFT")
    c.reset()
    s = c.update(det_at(int(W * 0.15), H // 2))
    check("move is TURN_LEFT", s.move is ChassisMove.TURN_LEFT, s.message)
    c.reset()
    s = c.update(det_at(int(W * 0.15), int(H * 0.15)))
    check("move is FORWARD_LEFT", s.move is ChassisMove.FORWARD_LEFT, s.message)

    print("\n[F] alignment streak resets when tin moves away")
    c.reset()
    for _ in range(3):
        c.update(det_at(W // 2, H // 2))
    c.update(det_at(int(W * 0.1), H // 2))          # breaks the streak
    s = c.update(det_at(W // 2, H // 2))
    check("stable requires a fresh streak", not s.stable)

    print("\n[G] mirror_x flips left/right classification")
    cm = IBVSCentering(CenteringConfig(ema_alpha=0.0, mirror_x=-1))
    s = cm.update(det_at(int(W * 0.15), H // 2))
    check("mirrored: left pixel -> TURN_RIGHT", s.move is ChassisMove.TURN_RIGHT, s.message)

    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: all offline checks PASSED")
    sys.exit(0)


def live():
    import cv2
    from src.perception.detector import AluminiumCanDetector

    detector = AluminiumCanDetector(device="cpu")  # Pi has no CUDA
    detector.start()
    centering = IBVSCentering()
    print("\nLive centering — press 'q' in the video window (or Ctrl+C) to stop.")
    print("To calibrate the sweet spot: put a tin at the arm's best grasp point,")
    print("read center=(px,py) below, set CenteringConfig.target_x = px/width,")
    print("target_y = py/height.\n")
    show = True  # auto-disabled below if there's no display (headless SSH)
    try:
        while True:
            result = detector.detect()
            status = centering.update(result)
            px = f"center={status.target_px}" if status.target_px else "center=none"
            print(f"{px:>22}  err=({status.error_x:+.2f},{status.error_y:+.2f})  "
                  f"move={status.move.value:<14} stable={status.stable}  | {status.message}")
            if show:
                frame = detector.get_annotated_frame(result)
                if frame is not None:
                    try:
                        cv2.imshow("IBVS centering (press q to quit)", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            break
                    except cv2.error:
                        print("[!] no display available — continuing text-only "
                              "(run on the Pi's desktop, not headless SSH, to see video)")
                        show = False
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        detector.stop()
        cv2.destroyAllWindows()


def _action_label(status) -> str:
    """Map a CenteringStatus to a short action word for the overlay."""
    if status.stable:
        return "CATCH"
    words = {
        ChassisMove.FORWARD: "FORWARD",
        ChassisMove.BACKWARD: "BACKWARD",
        ChassisMove.TURN_LEFT: "LEFT",
        ChassisMove.TURN_RIGHT: "RIGHT",
        ChassisMove.FORWARD_LEFT: "FORWARD + LEFT",
        ChassisMove.FORWARD_RIGHT: "FORWARD + RIGHT",
        ChassisMove.BACKWARD_LEFT: "BACKWARD + LEFT",
        ChassisMove.BACKWARD_RIGHT: "BACKWARD + RIGHT",
        ChassisMove.HOLD: "HOLD (aligning...)",
        ChassisMove.SEARCH: "SEARCH",
    }
    return words.get(status.move, status.move.value.upper())


def sim():
    """Real camera feed, FAKE detection (no YOLO). You place a simulated upright
    tin with the mouse (click or drag) and it STAYS PUT — like a real tin sitting
    on the floor. The suggested chassis action is drawn bottom-left, relative to
    the calibrated sweet spot. Run on the Pi desktop (needs a display)."""
    import cv2
    from src.perception.detector import open_camera_capture

    cap, fw, fh, fps = open_camera_capture(0, W, H)
    print(f"✓ Camera opened: {fw}x{fh} @ {fps:.0f}FPS  (SIM — fake tin, no YOLO)")
    centering = IBVSCentering()  # loads the calibrated sweet spot from the YAML
    # Simulated upright tin. Its size scales with depth: lower in the frame =
    # closer to the robot = bigger (near=big, far=small). Size is purely cosmetic
    # — centering tracks the base point, which is size-independent.
    ASPECT = 0.38            # width / height of an upright tin
    H_FAR, H_NEAR = 90, 280  # tin height in px when far (top) vs near (bottom)
    lo_y, hi_y = fh * 0.12, fh - 3.0

    # The tin's base point (where it meets the floor). A real tin doesn't move on
    # its own — you reposition it with the mouse and it stays there.
    pos = {"bx": fw / 2.0, "by": (lo_y + hi_y) / 2.0}

    def on_mouse(event, x, y, flags, param):
        dragging = event == cv2.EVENT_LBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON))
        if dragging:
            pos["bx"] = float(x)
            pos["by"] = float(min(max(y, lo_y), hi_y))

    win = "IBVS sim  (click/drag to place tin, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("\nClick or drag in the window to place the fake tin (it stays put).")
    print("q = quit.\n")
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠ failed to read frame from camera")
                break

            bx, by = pos["bx"], pos["by"]
            # Size from depth: lower in the frame (larger by) = nearer = bigger.
            t = (by - lo_y) / (hi_y - lo_y)          # 0 at top/far .. 1 at bottom/near
            th = int(H_FAR + t * (H_NEAR - H_FAR))
            tw = int(th * ASPECT)
            bx = min(max(bx, tw / 2.0), fw - tw / 2.0)

            ibx, iby = int(bx), int(by)
            x1, y1, x2, y2 = ibx - tw // 2, iby - th, ibx + tw // 2, iby
            box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.99)
            result = DetectionResult(detections=[box], frame_width=fw, frame_height=fh)
            status = centering.update(result)

            # Sweet spot (target) from the loaded config — yellow cross.
            tx = int(centering.config.target_x * fw)
            ty = int(centering.config.target_y * fh)
            cv2.drawMarker(frame, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(frame, "sweet spot", (tx + 10, ty - 10), font, 0.5, (0, 255, 255), 1)

            # Fake tin box (green) + tracked base point (red dot).
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "fake tin", (x1, max(y1 - 8, 14)), font, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (ibx, iby), 6, (0, 0, 255), -1)
            cv2.putText(frame, "click/drag to move", (x1, min(y2 + 20, fh - 8)),
                        font, 0.45, (210, 210, 210), 1)

            # Suggested action, bottom-left (black outline + colored fill).
            action = _action_label(status)
            color = (0, 255, 0) if status.stable else (0, 165, 255)
            cv2.putText(frame, action, (20, fh - 25), font, 1.2, (0, 0, 0), 6)
            cv2.putText(frame, action, (20, fh - 25), font, 1.2, color, 2)

            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--live" in sys.argv:
        live()
    elif "--sim" in sys.argv:
        sim()
    else:
        offline()
