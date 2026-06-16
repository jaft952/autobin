"""
Verification for IBVS Stage-1 chassis centering (src/visual_servoing/ibvs_centering.py).

Two modes:
  python tests/test_ibvs_centering.py
      OFFLINE logic checks with synthetic detections — no camera, no YOLO model,
      runs anywhere. Exit code 0 = all passed.

  python tests/test_ibvs_centering.py --live
      LIVE mode on the Pi: opens the camera + YOLO (on CPU), prints the centering
      status every frame, and prints the tin's pixel center so you can calibrate
      the sweet spot (CenteringConfig.target_x / target_y).
      Put a tin at the arm's comfortable grasp position, read the printed
      "center=(px,py)", divide by the frame size, and set target_x/y to that.
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
    """Synthetic single-tin DetectionResult centered at pixel (cx, cy)."""
    box = BoundingBox(x1=cx - 40, y1=cy - 60, x2=cx + 40, y2=cy + 60, confidence=0.9)
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


if __name__ == "__main__":
    if "--live" in sys.argv:
        live()
    else:
        offline()
