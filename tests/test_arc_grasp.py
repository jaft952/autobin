"""
── REPL commands ───────────────────────────────────────────────────────────
  c2 145        set CH2 to 145              (c1..c5 = servos, c6 = gripper)
  c1 +2         nudge CH1 by +2 degrees     (also -2 etc.)
  96.7 96.7 100 100 90    set all five CH1..CH5 at once
  p             adopt a saved grasp pose (middle sample of the first row)
  o / c         gripper open / close
  h             home
  pose          print the current pose
  st            print calibration status
  cam           just look through the camera (click shows coords; q closes)
  y             NEW ARC ROW: camera pops up -> click tin base -> s
                (starts a row at that pixel-y with the CURRENT CH1..CH5 as
                 its first — middle — sample)
  a             ADD SAMPLE to the nearest row: camera pops up -> click -> s
                (stores clicked pixel-x + CURRENT CH1..CH5 in that row)
  g             GRAB TEST: camera pops up -> click the tin -> s -> arm grabs
  q             quit

"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.arc_grasp import (
    ArcGraspSolver, save_config, NY_TOL_DEFAULT, NX_TOL_DEFAULT,
)
from src.hardware.actuators.pca9685_driver import ArmActuator, stepped_move


W, H = 1920, 1080


HOME_ARM = [96.7, 96.7, 150.0, 20.0, 90.0]
LIFT_ARM = [96.7, 96.7, 100.0, 100.0, 90.0]
GRIPPER_OPEN, GRIPPER_CLOSE = 0.0, 40.0
STEP_DEG, STEP_DELAY = 2.0, 0.5      



# Safe grab order (channel indices): CH1 base -> CH5 roll -> CH3 elbow ->
# CH4 wrist -> CH2 shoulder (LAST = final descent onto the tin). CH5 is set
# EARLY, while the arm is still lifted, so a stale roll from a previous grab
# (e.g. 180 from a lying tin) is corrected before anything comes near the
# floor. Gripper (CH6) closes after this, then CH2 lifts back up.
GRAB_APPROACH_ORDER = [0, 4, 2, 3, 1]

CH_NAMES = ["CH1 base", "CH2 shoulder", "CH3 elbow", "CH4 wrist", "CH5 roll", "CH6 grip"]


class Arm:
    """Tracks pose and moves ONE channel at a time, gently.

    Optionally holds a wheel brake: while the arm descends/grabs it shakes
    the chassis, and coasting wheels let the robot creep off the calibrated
    spot. brake_wheels() shorts the motor windings so the base resists that
    push; it's best-effort (no motor driver wired -> just skipped)."""

    def __init__(self):
        self.act = ArmActuator()
        self.arm = list(HOME_ARM)
        self.gripper = GRIPPER_OPEN
        # Best-effort wheel brake: the motor driver may not be wired on the
        # calibration bench, so a failure here must not kill the arm tool.
        self.wheels = None
        try:
            from src.hardware.actuators.pwm_driver import PWMActuator
            self.wheels = PWMActuator()
            print("[wheels] brake available — held during grabs.")
        except Exception as exc:
            print(f"[wheels] no motor driver ({exc}); grabs run without brake.")
        # NOTHING moves on startup (calibration tools must never surprise-
        # move the arm; same policy as servo_jog.py). The tracked pose is
        # ASSUMED to be home — if the arm isn't actually there, press 'h'
        # first: it force-commands home regardless of the tracked pose.
        print("[arm] startup: no movement. Tracked pose assumes HOME — press "
              "'h' to force-home if the arm isn't actually there.")

    def force_home(self):
        """Command every servo to HOME + open gripper, ignoring the tracked
        pose. With no feedback we ASSERT a known state instead of assuming one.
        Used by the 'h' command (so 'h' always responds, even when the tracked
        pose already says 'home')."""
        print("[arm] force-home: commanding all channels to home.")
        self.act.set_arm_angles(HOME_ARM)
        self.act.set_gripper_angle(GRIPPER_OPEN)
        self.arm = list(HOME_ARM)
        self.gripper = GRIPPER_OPEN

    def brake_wheels(self):
        if self.wheels is not None:
            self.wheels.brake()

    def release_wheels(self):
        if self.wheels is not None:
            self.wheels.stop()

    def _one(self, ch, value):
        start = list(self.arm) + [self.gripper]
        target = list(start)
        target[ch] = max(0.0, min(180.0, float(value)))
        stepped_move(self.act, start, target, STEP_DEG, STEP_DELAY)
        # WRITE-THROUGH guarantee: the commanded channel ALWAYS receives its
        # target, even when the tracked pose claims it's already there.
        # There is no joint feedback, so tracking can be wrong (e.g. after
        # boot) — the old diff-only path silently dropped such commands
        # ('c1 96.7' did nothing because tracking said 96.7 already).
        # A duplicate write is harmless; a dropped command is not.
        self.act.set_channel_angle(ch, target[ch])
        if ch < 5:
            self.arm[ch] = target[ch]
        else:
            self.gripper = target[ch]

    def goto(self, arm5, label=""):
        if label:
            print(f"[arm] {label}")
        for ch in range(5):
            self._one(ch, arm5[ch])   # no diff-skip: every channel is asserted

    def set_gripper(self, v):
        self._one(5, v)

    def print_pose(self):
        angles = "  ".join(f"{n.split()[0]}={v:.1f}" for n, v in zip(CH_NAMES, self.arm))
        print(f"[pose] {angles}  grip={self.gripper:.1f}")

    def grab(self, solved):
        """solved = [CH1..CH5]. Safe ordered grab so nothing sweeps the floor:

          1. lift to a high/folded pose (CH2/CH3/CH4) at the current azimuth
          2. approach in order CH1 base -> CH5 roll -> CH3 -> CH4 -> CH2
             (shoulder LAST = the only channel that lowers onto the tin), so
             the wrist ROLL is set while the arm is still high — a stale roll
             from a previous (e.g. lying) grab can no longer hit the ground
          3. close gripper (CH6), then lift the shoulder back up, holding

        Wheels are braked the whole time so the shaking can't drift the base."""
        print(f"[grab] pose CH1-5 = {solved}")
        self.brake_wheels()
        try:
            self.set_gripper(GRIPPER_OPEN)
            for ch in (1, 2, 3):                 # CH2, CH3, CH4 -> lift high
                self._one(ch, LIFT_ARM[ch])
            for ch in GRAB_APPROACH_ORDER:       # CH1, CH5, CH3, CH4, CH2(down)
                self._one(ch, solved[ch])
            self.set_gripper(GRIPPER_CLOSE)      # CH6 close on the tin
            self._one(1, LIFT_ARM[1])            # lift shoulder back up, holding
        finally:
            self.release_wheels()
        print("[grab] done — holding. o = release, h = home.")


class Camera:
    """Opened once, read only inside click_point() — no continuous streaming."""

    def __init__(self):
        from src.perception.detector import open_camera_capture
        self.cap, self.fw, self.fh, _fps = open_camera_capture(0, W, H)
        print(f"✓ camera {self.fw}x{self.fh}")

    def click_point(self, rows, title, sticky=False):
        """Popup window: click the tin, 's'/Enter accepts, 'q'/Esc cancels.
        rows = the ACTIVE domain's calibrated rows (drawn as overlay).
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
                for r in rows:                 # calibrated arc rows
                    yy = int(float(r["ny"]) * self.fh)
                    cv2.line(frame, (0, yy), (self.fw, yy), (0, 200, 200), 1)
                    for s in r["samples"]:     # sampled azimuth points on the arc
                        xx = int(float(s["nx"]) * self.fw)
                        cv2.drawMarker(frame, (xx, yy), (0, 200, 200),
                                       cv2.MARKER_DIAMOND, 10, 1)
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
    pose = "upright"            # active calibration domain: 'lie'/'stand' to switch
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

    def save_and_reload():
        save_config(cfg)
        solver.reload()
        print(f"[cfg] saved. {solver.status()}")

    print("\nCommands: c2 145 | c1 +2 | 5 angles | p o c h pose st | cam y a g | brake coast | q")
    print("Domains:  lie = calibrate LYING grid, stand = back to UPRIGHT grid.")
    print("          (place the tin in that pose, jog until the grab works, then y/a)")
    print("Wheels:   brake = hold the base (test by pushing it), coast = release.")
    print("Full walkthrough: header of this file.\n")

    while True:
        try:
            line = input(f"arc[{pose}]> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line == "q":
            break
        elif line == "brake":
            arm.brake_wheels()
            print("[wheels] BRAKED — try pushing the base; it should resist. "
                  "'coast' to release.")
        elif line == "coast":
            arm.release_wheels()
            print("[wheels] released (free to roll).")
        elif line == "lie":
            pose = "lying"
            print("[cfg] calibrating the LYING grid now. Lay the tin DOWN "
                  "pointing at the robot (straight); CH5 you save here is the "
                  "baseline — at runtime it's re-computed from the tin's angle.")
            print("[cfg] IMPORTANT: click the tin's MIDDLE (silhouette center), "
                  "NOT the base — that's what runtime feeds the solver for "
                  "lying tins (the bottom edge drifts with orientation).")
        elif line == "stand":
            pose = "upright"
            print("[cfg] calibrating the UPRIGHT grid now. Click the tin's "
                  "BASE (bottom-center, where it meets the floor).")
        elif line == "h":
            arm.force_home()          # force: always re-homes, even if tracking thinks it's home
        elif line == "o":
            arm.set_gripper(GRIPPER_OPEN)
        elif line == "c":
            arm.set_gripper(GRIPPER_CLOSE)
        elif line == "p":
            rows = solver.rows_for(pose)
            if not rows:
                print(f"no calibrated {pose} rows yet — jog manually, then 'y' to start one.")
                continue
            ss = rows[0]["samples"]
            mid = ss[len(ss) // 2]
            arm.goto([arm.arm[0]] + [float(v) for v in mid["arm"][1:]],
                     f"{pose} grasp pose (CH1 stays)")
        elif line == "pose":
            arm.print_pose()
        elif line == "st":
            print(f"[cfg] {solver.status()}")
        elif line == "cam":
            if camera:
                camera.click_point(solver.rows_for(pose), f"viewing only ({pose})", sticky=True)
        elif line == "y":
            if not camera:
                print("no camera.")
                continue
            where = "BASE" if pose == "upright" else "MIDDLE"
            pt = camera.click_point(solver.rows_for(pose),
                                    f"NEW {pose.upper()} ARC: click tin {where} at the grasp pose")
            if pt is None:
                print("cancelled.")
                continue
            row = {
                "ny": round(pt[1], 4),
                "ny_tol": NY_TOL_DEFAULT,
                "nx_tol": NX_TOL_DEFAULT,
                "samples": [{"nx": round(pt[0], 4),
                             "arm": [round(v, 1) for v in arm.arm]}],
            }
            cfg[pose].setdefault("rows", []).append(row)
            print(f"[cal] new {pose} arc row: ny={row['ny']}, first sample "
                  f"nx={row['samples'][0]['nx']} arm={row['samples'][0]['arm']}")
            save_and_reload()
        elif line == "a":
            if not camera:
                print("no camera.")
                continue
            if not cfg[pose].get("rows"):
                print(f"no {pose} arc rows yet — 'y' first.")
                continue
            where = "BASE" if pose == "upright" else "MIDDLE"
            pt = camera.click_point(solver.rows_for(pose),
                                    f"ADD {pose.upper()} SAMPLE: click tin {where} (CH1={arm.arm[0]:.1f})")
            if pt is None:
                print("cancelled.")
                continue
            # attach to the row whose ny is closest to the clicked pixel-y
            row = min(cfg[pose]["rows"], key=lambda r: abs(float(r["ny"]) - pt[1]))
            gap = abs(float(row["ny"]) - pt[1])
            if gap > 2 * float(row.get("ny_tol", NY_TOL_DEFAULT)):
                print(f"[cal] WARNING: clicked ny={pt[1]:.3f} is far from the nearest "
                      f"row (ny={float(row['ny']):.3f}) — same arc? Saving anyway; "
                      f"'y' instead if this was a NEW distance.")
            row.setdefault("samples", []).append(
                {"nx": round(pt[0], 4), "arm": [round(v, 1) for v in arm.arm]})
            print(f"[cal] sample added to row ny={float(row['ny']):.3f}: "
                  f"nx={pt[0]:.4f} arm={[round(v, 1) for v in arm.arm]}")
            save_and_reload()
        elif line == "g":
            if not camera:
                print("no camera.")
                continue
            where = "BASE" if pose == "upright" else "MIDDLE"
            pt = camera.click_point(solver.rows_for(pose),
                                    f"GRAB TEST ({pose}): click the tin {where}")
            if pt is None:
                print("cancelled.")
                continue
            # No live angle in this click-based tool: a lying grab test
            # assumes the tin is placed STRAIGHT at the robot (angle 90).
            # For angle-driven CH5 with real detections use test_arc_live.py.
            solved = solver.solve(pt[0], pt[1], pose=pose)
            if solved is None:
                print(f"[solve] ({pt[0]:.3f},{pt[1]:.3f}) NOT grabbable in {pose} "
                      f"(outside band / CH1 range / not calibrated). {solver.status()}")
                continue
            if pose == "lying":
                print("[grab] lying test assumes tin points AT the robot "
                      f"(CH5={solved[4]:.0f}); angled tins: test_arc_live.py")
            arm.grab(solved)
        elif handle_servo_command(arm, line):
            pass
        else:
            print("unknown. commands: c2 145 | c1 +2 | 5 angles | p o c h pose st | cam y a g | lie stand | brake coast | q")

    arm.release_wheels()          # never leave the base braked after exit
    print("bye")


if __name__ == "__main__":
    main()
