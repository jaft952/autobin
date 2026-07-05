"""
tests/test_arc_grasp.py

Calibrate + live-test the ARC GRASP (empirical polar IK, src/arm/arc_grasp.py).
Terminal REPL drives everything; the camera window ONLY pops up at the moment a
pixel is needed (y / a / g / cam), so typing and servo moves never fight the
video loop. Run on the Pi desktop.

── REPL commands ───────────────────────────────────────────────────────────
  c2 145        set CH2 to 145              (c1..c5 = servos, c6 = gripper)
  c1 +2         nudge CH1 by +2 degrees     (also -2 etc.)
  96.7 96.7 100 100 90    set all five CH1..CH5 at once
  p             adopt the saved grasp pose (CH2..CH5 from table; CH1 stays)
  o / c         gripper open / close
  h             home
  pose          print the current pose
  st            print calibration status
  cam           just look through the camera (click shows coords; q closes)
  y             save RADIUS row: camera pops up -> click tin base -> s
                (stores clicked pixel-y + CURRENT CH2..CH5 pose)
  a             save AZIMUTH sample: camera pops up -> click tin base -> s
                (stores clicked pixel-x + CURRENT CH1; >=2 samples auto-fit)
  f             re-fit azimuth from all samples + save
  g             GRAB TEST: camera pops up -> click the tin -> s -> arm grabs
  q             quit

── Calibration session ─────────────────────────────────────────────────────
 1. RADIUS ROW: tin at the comfortable grasp spot. `p`, then fine-tune with
    `c2 +2` style commands until a grab (c then o) works. Then `y` and click
    the tin's base in the popup.
 2. AZIMUTH x3: move the tin LEFT along the same-distance arc. Jog `c1 +2`/
    `c1 -2` until the gripper sits right above it (verify with c/o). Then `a`
    and click the tin. Repeat center + right. Fit auto-saves.
 3. TEST: tin anywhere on the arc -> `g` -> click it -> arm grabs it.

Config: src/visual_servoing/config/centering_config.yaml, under its own
`arc_grasp:` key — saving PRESERVES every other key in that yaml (target_x,
lying_target_y, ...). Hand-editable; add a 2nd radius row with a near/far
pose to unlock radius interpolation.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.arc_grasp import ArcGraspSolver, fit_azimuth, save_config
from src.hardware.actuators.pca9685_driver import ArmActuator, stepped_move

W, H = 1280, 720

# Poses/values shared with tests/test_ibvs_centering.py's arm controller.
HOME_ARM = [96.7, 96.7, 150.0, 20.0, 90.0]
LIFT_ARM = [96.7, 96.7, 100.0, 100.0, 90.0]
GRIPPER_OPEN, GRIPPER_CLOSE = 0.0, 40.0
STEP_DEG, STEP_DELAY = 2.0, 0.5      # gentle single-channel moves (weak supply)

# Descend order for the grab: elbow/wrist first, shoulder (CH2) LAST so the
# arm drops onto the can at the end — mirrors the tuned GRASP_SEQUENCE idea.
DESCEND_ORDER = [2, 3, 4, 1]         # indices into [CH1..CH5]

CH_NAMES = ["CH1 base", "CH2 shoulder", "CH3 elbow", "CH4 wrist", "CH5 roll", "CH6 grip"]


class Arm:
    """Tracks pose and moves ONE channel at a time, gently."""

    def __init__(self):
        self.act = ArmActuator()
        self.arm = list(HOME_ARM)
        self.gripper = GRIPPER_OPEN
        self.goto(HOME_ARM, label="startup home")

    def _one(self, ch, value):
        start = list(self.arm) + [self.gripper]
        target = list(start)
        target[ch] = max(0.0, min(180.0, float(value)))
        stepped_move(self.act, start, target, STEP_DEG, STEP_DELAY)
        if ch < 5:
            self.arm[ch] = target[ch]
        else:
            self.gripper = target[ch]

    def goto(self, arm5, label=""):
        if label:
            print(f"[arm] {label}")
        for ch in range(5):
            if abs(arm5[ch] - self.arm[ch]) > 1e-9:
                self._one(ch, arm5[ch])

    def set_gripper(self, v):
        self._one(5, v)

    def print_pose(self):
        angles = "  ".join(f"{n.split()[0]}={v:.1f}" for n, v in zip(CH_NAMES, self.arm))
        print(f"[pose] {angles}  grip={self.gripper:.1f}")

    def grab(self, solved):
        """solved = [CH1..CH5]. Open, swing, descend, close, lift."""
        print(f"[grab] pose CH1-5 = {solved}")
        self.set_gripper(GRIPPER_OPEN)
        self.goto([solved[0]] + LIFT_ARM[1:], "swing to azimuth (lifted)")
        for ch in DESCEND_ORDER:
            self._one(ch, solved[ch])
        self.set_gripper(GRIPPER_CLOSE)
        self._one(1, LIFT_ARM[1])            # lift shoulder back up, holding
        print("[grab] done — holding. o = release, h = home.")


class Camera:
    """Opened once, read only inside click_point() — no continuous streaming."""

    def __init__(self):
        from src.perception.detector import open_camera_capture
        self.cap, self.fw, self.fh, _fps = open_camera_capture(0, W, H)
        print(f"✓ camera {self.fw}x{self.fh}")

    def click_point(self, solver, title, sticky=False):
        """Popup window: click the tin, 's'/Enter accepts, 'q'/Esc cancels.
        sticky=True keeps the window open (just looking). Returns (nx, ny) or None."""
        import cv2
        for _ in range(5):                     # flush stale buffered frames
            self.cap.read()
        state = {"click": None}

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["click"] = (x, y)
                print(f"  clicked ({x},{y})  norm=({x / self.fw:.3f},{y / self.fh:.3f})")

        win = f"{title}   (click tin base | s=accept | q=cancel)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        result = None
        font = cv2.FONT_HERSHEY_SIMPLEX
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("  camera read failed")
                    break
                for r in solver.radii:         # calibrated radius rows
                    yy = int(float(r["ny"]) * self.fh)
                    cv2.line(frame, (0, yy), (self.fw, yy), (0, 200, 200), 1)
                if state["click"]:
                    cv2.drawMarker(frame, state["click"], (0, 0, 255),
                                   cv2.MARKER_TILTED_CROSS, 24, 2)
                cv2.putText(frame, title, (10, 30), font, 0.7, (0, 255, 255), 2)
                cv2.imshow(win, frame)
                k = cv2.waitKey(30) & 0xFF
                if k in (ord("s"), 13, 32) and state["click"] and not sticky:
                    x, y = state["click"]
                    result = (x / self.fw, y / self.fh)
                    break
                if k in (ord("q"), 27):
                    break
        finally:
            cv2.destroyAllWindows()
        return result


def handle_servo_command(arm: Arm, line: str) -> bool:
    """Parse 'c2 145' / 'c1 +2' / five angles. Returns True if it was one."""
    parts = line.split()
    try:
        if len(parts) == 2 and parts[0].startswith("c") and parts[0][1:].isdigit():
            ch = int(parts[0][1:])
            if not 1 <= ch <= 6:
                print("channel must be c1..c6 (c6 = gripper)")
                return True
            cur = arm.gripper if ch == 6 else arm.arm[ch - 1]
            val = parts[1]
            target = cur + float(val) if val[0] in "+-" else float(val)
            arm._one(ch - 1, target)
            arm.print_pose()
            return True
        if len(parts) == 5:
            arm.goto([float(p) for p in parts], "typed pose")
            arm.print_pose()
            return True
    except ValueError as e:
        print(f"parse error: {e}")
        return True
    return False


def main():
    solver = ArcGraspSolver()
    cfg = solver.cfg
    print(f"[cfg] {solver.status()}")

    try:
        arm = Arm()
    except Exception as e:
        print(f"arm not available ({e}) — check power/I2C.")
        return
    try:
        camera = Camera()
    except Exception as e:
        print(f"camera not available ({e}) — y/a/g/cam disabled.")
        camera = None

    def refit_and_save():
        fit = fit_azimuth(cfg["azimuth"].get("samples") or [])
        if fit:
            cfg["azimuth"]["nx_center"], cfg["azimuth"]["ch1_center"], \
                cfg["azimuth"]["ch1_per_nx"] = fit
            print(f"[fit] nx_center={fit[0]}  ch1_center={fit[1]}  ch1_per_nx={fit[2]}")
        save_config(cfg)
        solver.reload()
        print(f"[cfg] saved. {solver.status()}")

    print("\nCommands: c2 145 | c1 +2 | 5 angles | p o c h pose st | cam y a f g | q")
    print("Full walkthrough: header of this file.\n")

    while True:
        try:
            line = input("arc> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line == "q":
            break
        elif line == "h":
            arm.set_gripper(GRIPPER_OPEN)
            arm.goto(HOME_ARM, "home")
        elif line == "o":
            arm.set_gripper(GRIPPER_OPEN)
        elif line == "c":
            arm.set_gripper(GRIPPER_CLOSE)
        elif line == "p":
            row = solver.radii[0] if solver.radii else cfg["radii"][0]
            arm.goto([arm.arm[0]] + [float(v) for v in row["arm"]], "grasp pose")
        elif line == "pose":
            arm.print_pose()
        elif line == "st":
            print(f"[cfg] {solver.status()}")
        elif line == "f":
            refit_and_save()
        elif line == "cam":
            if camera:
                camera.click_point(solver, "viewing only", sticky=True)
        elif line == "y":
            if not camera:
                print("no camera.")
                continue
            pt = camera.click_point(solver, "RADIUS: click tin at the grasp pose")
            if pt is None:
                print("cancelled.")
                continue
            row = cfg["radii"][0]
            row["ny"] = round(pt[1], 4)
            row["arm"] = [round(v, 1) for v in arm.arm[1:5]]
            print(f"[cal] radius row: ny={row['ny']}  arm(CH2-5)={row['arm']}")
            save_config(cfg)
            solver.reload()
        elif line == "a":
            if not camera:
                print("no camera.")
                continue
            pt = camera.click_point(solver, f"AZIMUTH: click tin (CH1={arm.arm[0]:.1f})")
            if pt is None:
                print("cancelled.")
                continue
            cfg["azimuth"].setdefault("samples", []).append(
                [round(pt[0], 4), round(arm.arm[0], 1)])
            print(f"[cal] azimuth sample #{len(cfg['azimuth']['samples'])}: "
                  f"nx={pt[0]:.4f} CH1={arm.arm[0]:.1f}")
            refit_and_save()
        elif line == "g":
            if not camera:
                print("no camera.")
                continue
            pt = camera.click_point(solver, "GRAB TEST: click the tin")
            if pt is None:
                print("cancelled.")
                continue
            solved = solver.solve(pt[0], pt[1])
            if solved is None:
                print(f"[solve] ({pt[0]:.3f},{pt[1]:.3f}) NOT grabbable "
                      f"(outside band / CH1 range / not calibrated). {solver.status()}")
                continue
            arm.grab(solved)
        elif handle_servo_command(arm, line):
            pass
        else:
            print("unknown. commands: c2 145 | c1 +2 | 5 angles | p o c h pose st | cam y a f g | q")

    print("bye")


if __name__ == "__main__":
    main()
