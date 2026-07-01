"""
tests/test_ibvs_centering.py

Offline unit tests + live / sim demos for the IBVS centering layer.

Modes:
    python test_ibvs_centering.py           -> offline unit tests (no camera)
    python test_ibvs_centering.py --live    -> live camera + real YOLO
    python test_ibvs_centering.py --sim     -> real camera + fake draggable tin
    python test_ibvs_centering.py --live --drive   -> live + chassis motion ON
    python test_ibvs_centering.py --live --auto    -> live + auto grab ON

--- MOTION CHANGES vs OLD VERSION ---
v1: pulse_for_error() blocked the loop with time.sleep() -> 1s stutter.
v2: apply_for_error() ran continuously but used discrete TURN_LEFT/FORWARD
    states -> spun in place during YOLO's slow inference window -> overshoot.
v3 (this version): drive_toward_target(move, error_x, error_y) computes wheel
    speeds directly from the raw error signal, steering-wheel style. The base
    drives in a continuous ARC toward the tin — both wheels always turn
    forward, the curve tightens automatically as error_x shrinks, and the
    approach speed scales with error_y (distance). No sleep, no spin-in-place,
    no jerky pulsing. BACKWARD / SEARCH / HOLD still use discrete proportional
    control since steering doesn't apply to them.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.detector import BoundingBox, DetectionResult
from src.perception.orientation import Orientation
from src.visual_servoing.ibvs_centering import (
    ChassisMove,
    CenteringConfig,
    IBVSCentering,
)

W, H = 1280, 720
failures = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def det_at(cx, cy):
    """Synthetic single-tin DetectionResult whose BASE (ground-contact point —
    the bbox bottom-center that IBVSCentering tracks) is at pixel (cx, cy).
    The tin stands ~120px tall upward from that base."""
    box = BoundingBox(x1=cx - 40, y1=cy - 120, x2=cx + 40, y2=cy, confidence=0.5)
    return DetectionResult(detections=[box], frame_width=W, frame_height=H)


def det_lying_at(cx, cy, klass="lying"):
    """Synthetic LYING tin: a wide box whose CENTER (the point IBVSCentering tracks
    for a lying tin) is at pixel (cx, cy), tagged with a lying orientation."""
    box = BoundingBox(
        x1=cx - 100, y1=cy - 30, x2=cx + 100, y2=cy + 30,
        confidence=0.5,
        orientation=Orientation(angle=0.0, aspect=3.3, klass=klass),
    )
    return DetectionResult(detections=[box], frame_width=W, frame_height=H)


# ---------------------------------------------------------------------------
# Offline unit tests
# ---------------------------------------------------------------------------

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

    print("\n[E] turn-first: tin left -> TURN_LEFT; left+far still TURN_LEFT (aim before advancing)")
    c.reset()
    s = c.update(det_at(int(W * 0.15), H // 2))
    check("move is TURN_LEFT", s.move is ChassisMove.TURN_LEFT, s.message)
    c.reset()
    s = c.update(det_at(int(W * 0.15), int(H * 0.15)))
    check("left+far -> TURN_LEFT first (no diagonal)", s.move is ChassisMove.TURN_LEFT, s.message)

    print("\n[F] alignment streak resets when tin moves away")
    c.reset()
    for _ in range(3):
        c.update(det_at(W // 2, H // 2))
    c.update(det_at(int(W * 0.1), H // 2))   # breaks the streak
    s = c.update(det_at(W // 2, H // 2))
    check("stable requires a fresh streak", not s.stable)

    print("\n[G] mirror_x flips left/right classification")
    cm = IBVSCentering(CenteringConfig(ema_alpha=0.0, mirror_x=-1))
    s = cm.update(det_at(int(W * 0.15), H // 2))
    check("mirrored: left pixel -> TURN_RIGHT", s.move is ChassisMove.TURN_RIGHT, s.message)

    print("\n[H] lying tin tracks CENTER + lying sweet spot -> aligned when centered")
    cl = IBVSCentering(
        CenteringConfig(ema_alpha=0.0, lying_target_x=0.5, lying_target_y=0.5, lying_mode="point")
    )
    s = cl.update(det_lying_at(W // 2, H // 2))
    check("lying centered -> aligned", s.aligned, s.message)

    print("\n[I] lying X-off: POINT mode NOT aligned, LINE mode aligned (Y only)")
    det = det_lying_at(int(W * 0.2), H // 2)   # center X off, Y on the target
    sp = IBVSCentering(
        CenteringConfig(ema_alpha=0.0, lying_target_x=0.5, lying_target_y=0.5, lying_mode="point")
    ).update(det)
    check("lying POINT: X off -> NOT aligned", not sp.aligned, f"err=({sp.error_x},{sp.error_y})")
    sli = IBVSCentering(
        CenteringConfig(ema_alpha=0.0, lying_target_x=0.5, lying_target_y=0.5, lying_mode="line")
    ).update(det)
    check("lying LINE: X off but Y on -> aligned", sli.aligned, f"err=({sli.error_x},{sli.error_y})")

    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: all offline checks PASSED")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Live mode  (real camera + real YOLO + continuous steering motion)
# ---------------------------------------------------------------------------

def live(auto=False, drive=False):
    import cv2
    from src.perception.detector import AluminiumCanDetector

    IMGSZ = 640        # model was trained at 640 — keep matching inference size
    INFER_EVERY = 1     # run YOLO every frame; motors steer continuously between
                        # frames using the last known error, so no benefit to
                        # skipping. Raise to 2/3 only if Pi CPU is maxed out.

    # Minimum error magnitude before sending a move command.
    # Filters out tiny jitter errors near the sweet spot to prevent oscillation.
    MIN_ERROR_TO_MOVE = 0.01

    detector = AluminiumCanDetector(device="cpu", imgsz=IMGSZ)
    detector.start()
    centering = IBVSCentering()
    arm = _make_arm()
    chassis = _make_chassis()
    driving = drive and chassis is not None

    print(f"\nLive centering. grab={'AUTO' if auto else 'MANUAL'}, "
          f"drive={'ON' if driving else 'OFF'}.")
    print("Keys:  a = grab auto/manual,  m = drive on/off,  c = grab now,  "
          "h = home,  q = quit.")
    print("(Steering mode: continuous arcing motion toward the tin — speed and\n"
          " curve angle both scale with error, no spin-in-place, no pulsing.)\n")

    show = True
    armed = True
    result = None
    status = None
    i = 0

    try:
        while True:
            frame = detector.read_frame()
            if frame is None:
                continue

            if i % INFER_EVERY == 0:
                result = detector.infer(frame)
                status = centering.update(result)

                error_mag = max(abs(status.error_x), abs(status.error_y))

                px = f"center={status.target_px}" if status.target_px else "center=none"
                print(
                    f"{px:>22}  err=({status.error_x:+.2f},{status.error_y:+.2f})  "
                    f"move={status.move.value:<14} stable={status.stable}  "
                    f"err_mag={error_mag:.2f}  | {status.message}"
                )

                if driving:
                    if error_mag > MIN_ERROR_TO_MOVE:
                        # Steering mode: wheel speeds computed directly from
                        # (error_x, error_y) -> continuous arc, no spin-in-place.
                        chassis.drive_toward_target(
                            status.move, status.error_x, status.error_y
                        )
                    else:
                        # Very close to sweet spot: stop and let debounce settle
                        chassis.stop()

                if auto and arm is not None:
                    if status.stable and armed:
                        if driving:
                            chassis.stop()
                        print("[auto] CATCH -> grabbing")
                        arm.grasp(
                            result.best.orientation
                            if (result is not None and result.best)
                            else None
                        )
                        armed = False
                    elif not status.stable:
                        armed = True

            i += 1

            if not show:
                continue

            annotated = detector.get_annotated_frame(result) if result is not None else frame
            if status is not None:
                _draw_action_overlay(
                    annotated, status, arm is not None, auto,
                    driving if chassis is not None else None,
                )

            try:
                cv2.imshow(
                    "IBVS centering  (a=auto/manual  m=drive  c=grab  h=home  q=quit)",
                    annotated,
                )
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print(
                    "[!] no display available — continuing text-only "
                    "(run on the Pi's desktop, not headless SSH, to see video)"
                )
                show = False
                continue

            if key == ord("q"):
                break
            if key == ord("a"):
                auto = not auto
                armed = True
                print(f"[mode] grab = {'AUTO' if auto else 'MANUAL'}")
            if key == ord("m") and chassis is not None:
                driving = not driving
                if not driving:
                    chassis.stop()
                print(f"[mode] drive = {'ON' if driving else 'OFF'}")
            if key == ord("c") and arm is not None:
                arm.grasp(
                    result.best.orientation
                    if (result is not None and result.best)
                    else None
                )
            if key == ord("h") and arm is not None:
                arm.home()

    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        try:
            if driving and chassis is not None:
                chassis.stop()
        except Exception:
            pass
        try:
            detector.stop()
        except Exception:
            pass
        try:
            if chassis is not None:
                chassis.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

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


def _draw_action_overlay(frame, status, can_grab, auto=False, driving=None):
    import cv2
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    mode = "AUTO" if auto else "MANUAL"
    mcol = (0, 220, 0) if auto else (0, 165, 255)
    cv2.putText(frame, f"grab:{mode} (a=toggle)", (fw - 300, 32), font, 0.6, (0, 0, 0), 4)
    cv2.putText(frame, f"grab:{mode} (a=toggle)", (fw - 300, 32), font, 0.6, mcol, 1)

    if driving is not None:
        dmode = "ON" if driving else "OFF"
        dcol = (0, 220, 0) if driving else (0, 165, 255)
        cv2.putText(frame, f"drive:{dmode} (m=toggle)", (fw - 300, 58), font, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, f"drive:{dmode} (m=toggle)", (fw - 300, 58), font, 0.6, dcol, 1)

    action = _action_label(status)
    color = (0, 255, 0) if status.stable else (0, 165, 255)
    cv2.putText(frame, action, (20, fh - 25), font, 1.2, (0, 0, 0), 6)
    cv2.putText(frame, action, (20, fh - 25), font, 1.2, color, 2)

    if status.stable and can_grab:
        cue = "AUTO grabbing..." if auto else "press 'c' to grab"
        cv2.putText(frame, cue, (20, fh - 72), font, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, cue, (20, fh - 72), font, 0.7, (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# Arm controller
# ---------------------------------------------------------------------------

GRASP_ARM = [103.0, 145.0, 75.0, 168.0, 90.0]
BIN_ARM   = [96.7,  96.7,  100.0, 20.0,  90.0]
HOME_ARM  = [96.7,  96.7,  150.0, 20.0,  90.0]
LIFT_ARM  = [96.7,  96.7,  100.0, 100.0, 90.0]

GRASP_SEQUENCE = [(0, 100.0), (2, 20.0), (3, 30.0), (1, 170.0)]
GRIPPER_OPEN   = 0.0
GRIPPER_CLOSE  = 40.0
GRASP_STEP_DEG   = 2.0    # smaller = gentler move = lower peak current (weak supply)
GRASP_STEP_DELAY = 0.5    # longer = the supply recovers between steps

LYING_ARM            = [103.0, 167.0, 75.0, 150.0]
LYING_ROLL_REF_ANGLE = 0.0
LYING_ROLL_REF       = 180.0
LYING_ROLL_SIGN      = -1.0


class _ArmController:

    def __init__(self):
        from src.hardware.actuators.pca9685_driver import ArmActuator, stepped_move
        self.actuator = ArmActuator()
        self._step    = stepped_move
        self.arm      = list(HOME_ARM)
        self.gripper  = GRIPPER_OPEN

    def move_to(self, target_arm, target_gripper, label=""):
        # Move ONE servo at a time (CH1..CH5, then CH6 gripper) so only one motor
        # ever draws current at once — critical on a weak supply, where two servos
        # moving together sag the rail and drop the arm. Each channel steps gently.
        print(f"[grasp] {label}")
        for ch in range(5):
            if abs(target_arm[ch] - self.arm[ch]) > 1e-9:
                self._one_channel(ch, target_arm[ch])
        if abs(target_gripper - self.gripper) > 1e-9:
            self._one_channel(5, target_gripper)

    def _one_channel(self, ch, value):
        """Step just channel `ch` to `value` — only that servo moves, the rest hold."""
        start = list(self.arm) + [self.gripper]
        target = list(start)
        target[ch] = value
        self._step(self.actuator, start, target, GRASP_STEP_DEG, GRASP_STEP_DELAY)
        if ch < 5:
            self.arm[ch] = value
        else:
            self.gripper = float(value)

    def grasp(self, orientation=None):
        klass = getattr(orientation, "klass", None)
        if klass in ("lying", "axial"):
            self._grasp_lying(float(getattr(orientation, "angle", 0.0) or 0.0))
        else:
            self._grasp_upright()

    def _grasp_lying(self, angle):
        ch5 = max(
            0.0,
            min(180.0, LYING_ROLL_REF + LYING_ROLL_SIGN * (angle - LYING_ROLL_REF_ANGLE)),
        )
        print(f"[grasp] LYING pick-and-place (angle={angle:.0f} deg -> CH5={ch5:.0f})...")
        self.move_to(LIFT_ARM, GRIPPER_OPEN,  "ready pose")
        pose = LYING_ARM + [ch5]
        self.move_to(pose,     GRIPPER_OPEN,  "descend to lying pose")
        self.move_to(pose,     GRIPPER_CLOSE, "close on can")
        self.move_to(LIFT_ARM, GRIPPER_CLOSE, "lift (holding)")
        self.move_to(BIN_ARM,  GRIPPER_CLOSE, "move to bin (holding)")
        self.move_to(BIN_ARM,  GRIPPER_OPEN,  "release into bin")
        print("[grasp] done. press 'h' to return home.")

    def _grasp_upright(self):
        print("[grasp] UPRIGHT pick-and-place...")
        self.move_to(LIFT_ARM, GRIPPER_OPEN, "ready pose")
        self.move_to(self.arm, GRIPPER_OPEN, "open gripper")
        undo = []
        for ch, ang in GRASP_SEQUENCE:
            undo.append((ch, self.arm[ch]))
            target = list(self.arm)
            target[ch] = ang
            self.move_to(target, self.gripper, f"descend CH{ch + 1} -> {ang:.0f}")
        self.move_to(self.arm, GRIPPER_CLOSE, "close on can")
        for ch, prev in reversed(undo):
            target = list(self.arm)
            target[ch] = prev
            self.move_to(target, self.gripper, f"lift CH{ch + 1} -> {prev:.0f}")
        self.move_to(BIN_ARM, self.gripper,  "move to bin (holding)")
        self.move_to(BIN_ARM, GRIPPER_OPEN,  "release into bin")
        print("[grasp] done. press 'h' to return home.")

    def home(self):
        self.move_to(HOME_ARM, GRIPPER_OPEN, "home")


def _make_arm():
    try:
        arm = _ArmController()
        print("[grasp] arm ready — press 'c' to grab, 'h' to home.")
        return arm
    except Exception as exc:
        print(f"[grasp] arm not available ({exc}); 'c'/'h' grasp disabled.")
        return None


def _make_chassis():
    try:
        from src.visual_servoing.chassis_controller import ChassisController
        chassis = ChassisController()
        print("[drive] chassis ready — press 'm' to toggle motion on/off.")
        return chassis
    except Exception as exc:
        print(f"[drive] chassis not available ({exc}); motion disabled.")
        return None


# ---------------------------------------------------------------------------
# Sim mode  (real camera, fake draggable tin, no YOLO)
# ---------------------------------------------------------------------------

def sim(auto=False, drive=False):
    import cv2
    from src.perception.detector import open_camera_capture

    cap, fw, fh, fps = open_camera_capture(0, W, H)
    print(f"✓ Camera opened: {fw}x{fh} @ {fps:.0f}FPS  (SIM — fake tin, no YOLO)")
    centering = IBVSCentering()
    arm       = _make_arm()
    chassis   = _make_chassis()
    driving   = drive and chassis is not None
    armed     = True

    MIN_ERROR_TO_MOVE = 0.01

    ASPECT   = 0.38
    H_FAR, H_NEAR = 90, 280
    lo_y, hi_y = fh * 0.12, fh - 3.0

    pos = {"bx": fw / 2.0, "by": (lo_y + hi_y) / 2.0}

    def on_mouse(event, x, y, flags, param):
        dragging = event == cv2.EVENT_LBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)
        )
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
                print("warning: failed to read frame from camera")
                break

            bx, by = pos["bx"], pos["by"]
            t  = (by - lo_y) / (hi_y - lo_y)
            th = int(H_FAR + t * (H_NEAR - H_FAR))
            tw = int(th * ASPECT)
            bx = min(max(bx, tw / 2.0), fw - tw / 2.0)

            ibx, iby = int(bx), int(by)
            x1, y1, x2, y2 = ibx - tw // 2, iby - th, ibx + tw // 2, iby
            box    = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.5)
            result = DetectionResult(detections=[box], frame_width=fw, frame_height=fh)
            status = centering.update(result)

            error_mag = max(abs(status.error_x), abs(status.error_y))

            if driving:
                if error_mag > MIN_ERROR_TO_MOVE:
                    chassis.drive_toward_target(
                        status.move, status.error_x, status.error_y
                    )
                else:
                    chassis.stop()

            if auto and arm is not None:
                if status.stable and armed:
                    if driving:
                        chassis.stop()
                    print("[auto] CATCH -> grabbing")
                    arm.grasp(
                        result.best.orientation
                        if (result is not None and result.best)
                        else None
                    )
                    armed = False
                elif not status.stable:
                    armed = True

            tx = int(centering.config.target_x * fw)
            ty = int(centering.config.target_y * fh)
            cv2.drawMarker(frame, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(frame, "sweet spot", (tx + 10, ty - 10), font, 0.5, (0, 255, 255), 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "fake tin", (x1, max(y1 - 8, 14)), font, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (ibx, iby), 6, (0, 0, 255), -1)
            cv2.putText(
                frame, "click/drag to move",
                (x1, min(y2 + 20, fh - 8)), font, 0.45, (210, 210, 210), 1,
            )

            _draw_action_overlay(
                frame, status, arm is not None, auto,
                driving if chassis is not None else None,
            )

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto = not auto
                armed = True
                print(f"[mode] grab = {'AUTO' if auto else 'MANUAL'}")
            if key == ord("m") and chassis is not None:
                driving = not driving
                if not driving:
                    chassis.stop()
                print(f"[mode] drive = {'ON' if driving else 'OFF'}")
            if key == ord("c") and arm is not None:
                arm.grasp(
                    result.best.orientation
                    if (result is not None and result.best)
                    else None
                )
            if key == ord("h") and arm is not None:
                arm.home()

    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if driving and chassis is not None:
            chassis.stop()
        cap.release()
        if chassis is not None:
            chassis.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    auto  = "--auto"  in sys.argv
    drive = "--drive" in sys.argv
    if "--live" in sys.argv:
        live(auto, drive)
    elif "--sim" in sys.argv:
        sim(auto, drive)
    else:
        offline()