"""
Verification for IBVS Stage-1 chassis centering (src/visual_servoing/ibvs_centering.py).

Two modes:
  python tests/test_ibvs_centering.py
      OFFLINE logic checks with synthetic detections — no camera, no YOLO model,
      runs anywhere. Exit code 0 = all passed.

  python tests/test_ibvs_centering.py --live
      LIVE mode on the Pi: camera + YOLO (CPU). Shows the annotated video with the
      suggested action (FORWARD / BACKWARD / LEFT / RIGHT / CATCH ...) bottom-left.
      The action is only a SUGGESTION — grasping is MANUAL: press 'c' to grab,
      'h' to home, 'q' to quit. (Needs the arm/servos for c/h; vision still runs
      without them.)

  python tests/test_ibvs_centering.py --sim
      SIM mode on the Pi: real camera feed but NO YOLO. You place a fake "upright
      tin" with the mouse (click or drag) and it STAYS PUT, like a real tin on the
      floor — nothing drifts on its own. Same action overlay + manual grab as live
      (c = grab, h = home, q = quit) — a safe way to test the grasp without YOLO.
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

    # Pi 4 CPU perf: infer at a small image size, and run YOLO only every Nth
    # frame while showing the camera every frame — keeps the preview smooth and
    # keys responsive even though inference itself is slow. Tune if needed.
    IMGSZ = 640
    INFER_EVERY = 3
    detector = AluminiumCanDetector(device="cpu", imgsz=IMGSZ)  # Pi has no CUDA
    detector.start()
    centering = IBVSCentering()      # loads the calibrated sweet spot from the YAML
    arm = _make_arm()                # optional: enables 'c' grab / 'h' home
    print("\nLive centering. In the video window:  c = grab,  h = home,  q = quit.")
    print("(The action shown is only a SUGGESTION — YOU decide when to press 'c'.)\n")
    show = True       # auto-disabled below if there's no display (headless SSH)
    result = None     # last YOLO result, reused on the frames we don't infer
    status = None
    i = 0
    try:
        while True:
            frame = detector.read_frame()        # cheap: grab a frame, no YOLO
            if frame is None:
                continue
            if i % INFER_EVERY == 0:             # run YOLO only every Nth frame
                result = detector.infer(frame)
                status = centering.update(result)
                px = f"center={status.target_px}" if status.target_px else "center=none"
                print(f"{px:>22}  err=({status.error_x:+.2f},{status.error_y:+.2f})  "
                      f"move={status.move.value:<14} stable={status.stable}  | {status.message}")
            i += 1
            if not show:
                continue
            annotated = detector.get_annotated_frame(result) if result is not None else frame
            if status is not None:
                _draw_action_overlay(annotated, status, arm is not None)
            try:
                cv2.imshow("IBVS centering  (c=grab  h=home  q=quit)", annotated)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print("[!] no display available — continuing text-only "
                      "(run on the Pi's desktop, not headless SSH, to see video)")
                show = False
                continue
            if key == ord("q"):
                break
            if key == ord("c") and arm is not None:
                arm.grasp()
            if key == ord("h") and arm is not None:
                arm.home()
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


# ── Manual grasp (CH1-5 arm + CH6 gripper), triggered by 'c' in --live / --sim ──
# "sweet point 3" — a hand-tuned grasp pose: CH1..CH5 arm angles, then gripper(CH6).
GRASP_ARM = [103.0, 145.0, 75.0, 168.0, 90.0]   # arm angles at the grasp point
BIN_ARM  = [96.7, 96.7, 100.0, 20.0, 90.0]      # arm pose over the bin (== grasp_planner.BIN_DROP_ANGLES)
HOME_ARM = [96.7, 96.7, 150.0, 20.0, 90.0]      # rest pose (== grasp_planner.HOME_ANGLES)
# "Straight up" transit pose, used before/after big moves so the arm doesn't drag
# or sweep low. These are the upright/neutral angles from kinematics.py
# (SERVO_NEUTRAL_CMD indices 1-5; its gripper value 80 is ignored here — gripper
# state is managed per phase in grasp()).
LIFT_ARM = [96.7, 96.7, 100.0, 100.0, 90.0]
GRIPPER_OPEN = 120.0      # CH6 open: before the grab, and to release at the bin
GRIPPER_CLOSE = 80.0     # CH6 closed on the tin (at the grasp pose)
GRASP_STEP_DEG = 5.0      # max degrees any servo moves per step (smaller = slower/smoother)
GRASP_STEP_DELAY = 0.15   # seconds paused between steps (bigger = slower)


class _ArmController:
    """Gentle manual arm control for the test. Moves servos to a target pose in
    small steps (<= GRASP_STEP_DEG each, pausing between) so nothing slams.

    Tracks the last-commanded pose in software and ASSUMES the arm starts at HOME,
    so have the arm at its home pose before launching — otherwise the first move
    may be bigger than one step."""

    def __init__(self):
        from src.hardware.actuators.pca9685_driver import ArmActuator
        self.actuator = ArmActuator()
        self.arm = list(HOME_ARM)      # assumed current arm pose (not commanded here)
        self.gripper = GRIPPER_OPEN    # assumed current gripper

    def move_to(self, target_arm, target_gripper, label=""):
        """Interpolate from the current pose to the target so no single servo
        jumps more than GRASP_STEP_DEG per step."""
        import time
        start_arm, start_grip = list(self.arm), self.gripper
        deltas = [abs(t - s) for t, s in zip(target_arm, start_arm)]
        deltas.append(abs(target_gripper - start_grip))
        steps = max(1, int((max(deltas) + GRASP_STEP_DEG - 1e-6) // GRASP_STEP_DEG))
        print(f"[grasp] {label}: {steps} step(s), <= {GRASP_STEP_DEG:.0f} deg each")
        for k in range(1, steps + 1):
            f = k / steps
            arm = [s + (t - s) * f for s, t in zip(start_arm, target_arm)]
            grip = start_grip + (target_gripper - start_grip) * f
            self.actuator.set_arm_angles(arm)
            self.actuator.set_gripper_angle(grip)
            time.sleep(GRASP_STEP_DELAY)
        self.arm, self.gripper = list(target_arm), float(target_gripper)

    def grasp(self):
        """Full gentle pick-and-place, all moves in <= GRASP_STEP_DEG steps.
        Raises to the upright transit pose (LIFT_ARM) before descending, before
        swinging to the bin, and after releasing, so the arm never drags low."""
        print("[grasp] pick-and-place...")
        self.move_to(LIFT_ARM, GRIPPER_OPEN, "raise upright (ready)")
        self.move_to(GRASP_ARM, GRIPPER_OPEN, "descend (gripper open)")
        self.move_to(GRASP_ARM, GRIPPER_CLOSE, "close on tin")
        self.move_to(LIFT_ARM, GRIPPER_CLOSE, "raise upright (carry)")
        self.move_to(BIN_ARM, GRIPPER_CLOSE, "move to bin (holding)")
        self.move_to(BIN_ARM, GRIPPER_OPEN, "release into bin")
        self.move_to(LIFT_ARM, GRIPPER_OPEN, "raise upright (done)")
        print("[grasp] done. press 'h' to return home.")

    def home(self):
        """Gently return to the home/rest pose with the gripper open."""
        self.move_to(HOME_ARM, GRIPPER_OPEN, "home")


def _make_arm():
    """Best-effort create the arm controller; return None (and explain) if the
    arm/servo hardware isn't available, so vision still runs without it."""
    try:
        arm = _ArmController()
        print("[grasp] arm ready — press 'c' to grab, 'h' to home.")
        return arm
    except Exception as exc:
        print(f"[grasp] arm not available ({exc}); 'c'/'h' grasp disabled.")
        return None


def _draw_action_overlay(frame, status, can_grab):
    """Draw the suggested action bottom-left; cue the manual grab when stable."""
    import cv2
    fh = frame.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    action = _action_label(status)
    color = (0, 255, 0) if status.stable else (0, 165, 255)
    cv2.putText(frame, action, (20, fh - 25), font, 1.2, (0, 0, 0), 6)
    cv2.putText(frame, action, (20, fh - 25), font, 1.2, color, 2)
    if status.stable and can_grab:
        cue = "press 'c' to grab"
        cv2.putText(frame, cue, (20, fh - 72), font, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, cue, (20, fh - 72), font, 0.7, (0, 255, 0), 2)


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
    arm = _make_arm()            # optional: enables 'c' grab / 'h' home
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

            # Suggested action + manual-grab cue, bottom-left.
            _draw_action_overlay(frame, status, arm is not None)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and arm is not None:
                arm.grasp()
            if key == ord("h") and arm is not None:
                arm.home()
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
