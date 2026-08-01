"""
Arc-grasp calibration tool. The arm driver lives in src/arm/grasp_planner.py —
this file is only the REPL + camera viewfinder around it.

── REPL commands ───────────────────────────────────────────────────────────
  c2 145        set CH2 to 145              (c1..c5 = servos, c6 = gripper)
  c1 +2         nudge CH1 by +2 degrees     (also -2 etc.)
  96.7 96.7 100 100 90    set all five CH1..CH5 at once
  p             adopt a saved grasp pose (middle sample of the first row)
  o / c         gripper open / close
  h             home
  pose          print the current pose
  st            print calibration status
  speed D S     retune move pacing: D degrees per S seconds
  grip O C      retune the open/close gripper angles (driver clamps them)
  cam           just look through the camera (YOLO marks the tin; q closes)
  y             NEW ARC ROW: camera pops up -> YOLO marks the tin's reference
                point (upright: bbox bottom-center, lying: bbox center — the
                SAME point the robot uses at runtime) -> press 's' to store it
                with the CURRENT CH1..CH5 as the row's first (middle) sample.
                Click the image only to OVERRIDE a bad detection ('x' clears).
  a             ADD SAMPLE to the nearest row: same auto-pick flow -> s
                (stores the point's nx AND ny + CURRENT CH1..CH5 — rows are
                 curves, the edges of an arc sit lower in the image).
                REPLACES any existing sample within nx_tol of the new one,
                so re-calibrating a spot overwrites instead of accumulating.
  ls            list rows + samples (with indices) of the active domain
  del R S       delete sample S of row R (see 'ls'; highest index first)
  g             GRAB TEST: camera pops up -> auto point -> s -> arm grabs
                (ends HOLDING so you can check the grip; 'b' to dump)
  b             drop the held tin into the onboard bin (bin pose + open)
  lie / stand   switch the calibration domain (LYING / UPRIGHT grid)
  brake / coast wheel brake on/off (hold the base while testing)
  r             release the arm (cut PWM, goes limp)
  q             quit
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.arc_grasp import (
    ArcGraspSolver, save_config, row_ny_at,
    NY_TOL_DEFAULT, NY_TOL_NEAR_DEFAULT, NX_TOL_DEFAULT,
)
from src.arm.arc_calibration import choose_row_for_sample
from src.arm.grasp_planner import GraspPlanner


class Camera:
    """Viewfinder for calibration points.

    AUTO mode (default): runs the SAME YOLO detector the robot uses at
    runtime and marks the tin's reference point with the SAME convention
    (upright -> bbox bottom-center, lying -> bbox center), so calibrated
    pixels are exactly what layer3 will feed the solver. Clicking overrides
    the detection. FALLBACK: no model -> plain capture + manual clicks.
    """

    def __init__(self, model_path=None):
        import time as _time
        self.detector = None
        self.cap = None
        try:
            from src.perception.detector import (
                AluminiumCanDetector, RUNTIME_MODEL_PATH,
            )
            if model_path is None:
                model_path = RUNTIME_MODEL_PATH   # calibrate with THE runtime model
            # Same resolution AND confidence as the runtime CameraSensor so
            # calibration and runtime literally share pixels + detections.
            self.detector = AluminiumCanDetector(model_path=model_path,
                                                 conf_threshold=0.8,
                                                 frame_width=1280, frame_height=720)
            self.detector.start()
            # The Brio returns None for the first few reads after opening.
            frame = None
            for _ in range(20):
                frame = self.detector.read_frame()
                if frame is not None:
                    break
                _time.sleep(0.1)
            if frame is None:
                raise RuntimeError("camera gave no frame after 2s")
            self.fh, self.fw = frame.shape[:2]
            print(f"camera+YOLO {self.fw}x{self.fh} (model {model_path}) — "
                  f"'s' stores the DETECTED point; click only to override.")
        except Exception as exc:
            if self.detector is not None:
                try:
                    self.detector.stop()
                except Exception:
                    pass
            self.detector = None
            print(f"[cam] detector unavailable ({exc}) — MANUAL CLICK MODE.")
            from src.perception.detector import open_camera_capture
            self.cap, self.fw, self.fh, _fps = open_camera_capture(0, 1280, 720)
            print(f"camera {self.fw}x{self.fh}")

    def stop(self):
        if self.detector is not None:
            self.detector.stop()
        elif self.cap is not None:
            self.cap.release()

    def _draw_rows(self, frame, rows):
        import cv2
        for r in rows:                     # calibrated arc CURVES
            default_ny = float(r["ny"])
            pts = sorted((float(s["nx"]), float(s.get("ny", default_ny)))
                         for s in r["samples"])
            px = [(int(x * self.fw), int(y * self.fh)) for x, y in pts]
            for p1, p2 in zip(px, px[1:]):
                cv2.line(frame, p1, p2, (0, 200, 200), 1)
            if len(px) == 1:               # single sample: short tick
                x0, y0 = px[0]
                cv2.line(frame, (x0 - 40, y0), (x0 + 40, y0), (0, 200, 200), 1)
            for p in px:
                cv2.drawMarker(frame, p, (0, 200, 200), cv2.MARKER_DIAMOND, 10, 1)

    def pick_point(self, rows, title, pose="upright", sticky=False):
        """Popup window; returns (nx, ny) or None.
        's'/Enter accepts the auto-detected point (or the manual click if one
        was made); click = manual override; 'x' clears the override;
        'q'/Esc cancels. sticky=True keeps the window open (just looking)."""
        import cv2
        state = {"click": None}

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["click"] = (x, y) # type: ignore
                print(f"  manual override ({x},{y})  "
                      f"norm=({x / self.fw:.3f},{y / self.fh:.3f})")

        mode = "s=DETECTED point | click=override | x=clear | q=cancel" \
            if self.detector else "click tin | s=accept | q=cancel"
        win = f"{title}   ({mode})"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        result = None
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.detector is None:          # flush stale buffered frames
            for _ in range(5):
                self.cap.read() # type: ignore

        try:
            i = 0
            det_res = None
            while True:
                if self.detector is not None:
                    frame = self.detector.read_frame()
                    if frame is None:
                        print("  camera read failed")
                        break
                    if i % 2 == 0:                     # infer every 2nd frame
                        det_res = self.detector.infer(frame)
                    i += 1
                    shown = self.detector.get_annotated_frame(det_res) \
                        if det_res is not None else frame
                    if shown is None:
                        shown = frame
                else:
                    ok, shown = self.cap.read() #type: ignore
                    if not ok:
                        print("  camera read failed")
                        break

                self._draw_rows(shown, rows)

                # Auto reference point: SAME convention as layer3 at runtime.
                auto_pt = None
                best = det_res.best if det_res is not None else None
                if best is not None:
                    auto_pt = best.base_center if pose == "upright" else best.center
                    cv2.circle(shown, auto_pt, 9, (0, 0, 255), 2)
                    cv2.putText(shown, "AUTO", (auto_pt[0] + 12, auto_pt[1] + 4),
                                font, 0.55, (0, 0, 255), 2)
                    o = best.orientation
                    if o is not None:
                        expect = "upright" if pose == "upright" else ("lying", "axial")
                        ok_pose = (o.klass == expect) if pose == "upright" \
                            else (o.klass in expect)
                        if not ok_pose:
                            cv2.putText(shown,
                                        f"! detected {o.klass}, calibrating {pose}",
                                        (10, 62), font, 0.7, (0, 140, 255), 2)

                if self.detector is None:
                    cv2.putText(shown, "MANUAL MODE - detector unavailable, "
                                "click the tin", (10, 62), font, 0.7,
                                (0, 0, 255), 2)
                if state["click"]:
                    cv2.drawMarker(shown, state["click"], (255, 0, 0),
                                   cv2.MARKER_TILTED_CROSS, 24, 2)
                cv2.putText(shown, title, (10, 30), font, 0.7, (0, 255, 255), 2)
                cv2.imshow(win, shown)
                k = cv2.waitKey(30) & 0xFF
                if k == ord("x"):
                    state["click"] = None
                if k in (ord("s"), 13, 32) and not sticky:
                    chosen = state["click"] or auto_pt
                    if chosen is None:
                        print("  no detection and no click yet — click the tin.")
                        continue
                    src = "manual" if state["click"] else "AUTO"
                    result = (chosen[0] / self.fw, chosen[1] / self.fh)
                    print(f"  stored {src} point ({chosen[0]},{chosen[1]})  "
                          f"norm=({result[0]:.3f},{result[1]:.3f})")
                    break
                if k in (ord("q"), 27):
                    break
        finally:
            cv2.destroyAllWindows()
        return result


def handle_servo_command(arm: GraspPlanner, line: str) -> bool:
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
            arm.move_channel(ch - 1, target)
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
    import argparse
    ap = argparse.ArgumentParser(description="Arc-grasp calibration tool")
    ap.add_argument("--model", default=None,
                    help="YOLO weights for auto point-pick; default = the "
                         "runtime model (RUNTIME_MODEL_PATH in detector.py); "
                         "falls back to manual clicks if it can't load")
    ap.add_argument("--step-deg", type=float, default=None,
                    help="move pacing: degrees per --step-delay (default: "
                         "the shared arm_settings profile)")
    ap.add_argument("--step-delay", type=float, default=None,
                    help="move pacing: seconds per --step-deg")
    args = ap.parse_args()

    solver = ArcGraspSolver()
    cfg = solver.cfg
    pose = "upright"            # active calibration domain: 'lie'/'stand' to switch
    print(f"[cfg] {solver.status()}")

    try:
        arm = GraspPlanner(release_on_start=True)
        arm.set_speed(args.step_deg, args.step_delay)
    except Exception as e:
        print(f"arm not available ({e}) — check power/I2C.")
        return
    try:
        from src.hardware.actuators.pwm_driver import PWMActuator
        wheels = PWMActuator()
        print("[wheels] brake available — held during grabs.")
    except Exception as e:
        wheels = None
        print(f"[wheels] no motor driver ({e}); grabs run without brake.")
    try:
        camera = Camera(args.model)
    except Exception as e:
        print(f"camera not available ({e}) — y/a/g/cam disabled.")
        camera = None

    def save_and_reload():
        save_config(cfg)
        solver.reload()
        print(f"[cfg] saved. {solver.status()}")

    print("\nCommands: c2 145 | c1 +2 | 5 angles | p o c h b pose st ls del | "
          "speed D S | grip O C | cam y a g | brake coast | r=release arm | q")
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
            if wheels is not None:
                wheels.brake()
                print("[wheels] BRAKED — try pushing the base; it should resist. "
                      "'coast' to release.")
            else:
                print("[wheels] no motor driver.")
        elif line == "coast":
            if wheels is not None:
                wheels.stop()
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
            arm.goto("home")          # asserts every channel, even if tracking says home
            arm.open_gripper()
        elif line == "r":
            arm.release()             # cut PWM — arm goes limp
        elif line == "o":
            arm.open_gripper()
        elif line == "c":
            arm.close_gripper()
        elif line == "b":
            if wheels is not None:
                wheels.brake()
            try:
                arm.dump_to_bin()
            finally:
                if wheels is not None:
                    wheels.stop()
        elif line.startswith("speed"):
            parts = line.split()
            try:
                arm.set_speed(float(parts[1]), float(parts[2]))
                print(f"[arm] pacing {arm.profile.step_deg}deg / "
                      f"{arm.profile.step_delay}s = {arm.profile.speed_dps:.1f} deg/s")
            except (IndexError, ValueError):
                print("usage: speed <deg> <seconds>")
        elif line.startswith("grip"):
            parts = line.split()
            try:
                arm.set_gripper_angles(float(parts[1]), float(parts[2]))
                print(f"[arm] gripper open={arm.profile.gripper_open:.1f} "
                      f"close={arm.profile.gripper_closed:.1f} (driver-clamped)")
            except (IndexError, ValueError):
                print("usage: grip <open_deg> <close_deg>")
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
        elif line == "ls":
            rows = cfg[pose].get("rows", [])
            if not rows:
                print(f"no {pose} rows.")
                continue
            for ri, r in enumerate(rows):
                r.setdefault("samples", []).sort(key=lambda s: float(s["nx"]))
                print(f"row {ri}: ny={r['ny']}  ny_tol={r.get('ny_tol', NY_TOL_DEFAULT)}  "
                      f"nx_tol={r.get('nx_tol', NX_TOL_DEFAULT)}")
                for si, s in enumerate(r["samples"]):
                    ny_txt = (f"ny={s['ny']}" if "ny" in s
                              else f"ny=({r['ny']} legacy-flat)")
                    print(f"   [{si}] nx={s['nx']}  {ny_txt}  arm={s['arm']}")
            print("delete with: del <row> <sample>   (highest index first!)")
        elif line.startswith("del "):
            parts = line.split()
            try:
                ri, si = int(parts[1]), int(parts[2])
                row = cfg[pose]["rows"][ri]
                row["samples"].sort(key=lambda s: float(s["nx"]))
                gone = row["samples"].pop(si)
                print(f"[cal] deleted row {ri} sample [{si}] nx={gone['nx']}")
                if not row["samples"]:
                    cfg[pose]["rows"].pop(ri)
                    print(f"[cal] row {ri} had no samples left — row removed.")
                save_and_reload()
            except (IndexError, ValueError, KeyError) as e:
                print(f"usage: del <row> <sample> — see 'ls' for indices ({e})")
        elif line == "cam":
            if camera:
                camera.pick_point(solver.rows_for(pose), f"viewing only ({pose})",
                                  pose=pose, sticky=True)
        elif line == "y":
            if not camera:
                print("no camera.")
                continue
            pt = camera.pick_point(solver.rows_for(pose),
                                   f"NEW {pose.upper()} ARC: tin at the grasp spot",
                                   pose=pose)
            if pt is None:
                print("cancelled.")
                continue
            row = {
                "ny": round(pt[1], 4),
                "ny_tol": NY_TOL_DEFAULT,            # margin ABOVE the arc (far)
                "ny_tol_near": NY_TOL_NEAR_DEFAULT,  # ~zero BELOW (closer = overshoot)
                "nx_tol": NX_TOL_DEFAULT,
                # samples carry their OWN ny: a constant-radius arc sits
                # lower in the image at the edges, so the row is a curve.
                "samples": [{"nx": round(pt[0], 4), "ny": round(pt[1], 4),
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
            pt = camera.pick_point(solver.rows_for(pose),
                                   f"ADD {pose.upper()} SAMPLE (CH1={arm.arm[0]:.1f})",
                                   pose=pose)
            if pt is None:
                print("cancelled.")
                continue
            row, pdist, note = choose_row_for_sample(
                cfg[pose]["rows"], arm.arm, pt[0], pt[1])
            print(f"[cal] attach by posture: CH2-4 distance {pdist:.0f} deg"
                  + (f" — {note}" if note else ""))
            gap = abs(row_ny_at(row, pt[0]) - pt[1])
            if gap > 2 * float(row.get("ny_tol", NY_TOL_DEFAULT)):
                print(f"[cal] note: clicked ny={pt[1]:.3f} is {gap:.3f} from this "
                      f"row's current curve — fine at an edge (arcs dip there).")
            # Re-calibrating the same spot must overwrite, not accumulate:
            # mixing legacy flat samples with new ny-carrying ones zigzags the curve.
            new_nx = round(pt[0], 4)
            tol = float(row.get("nx_tol", NX_TOL_DEFAULT))
            samples = row.setdefault("samples", [])
            stale = [s for s in samples if abs(float(s["nx"]) - new_nx) <= tol]
            for s in stale:
                samples.remove(s)
            if stale:
                print(f"[cal] replaced {len(stale)} old sample(s) within "
                      f"nx±{tol} of {new_nx} (old nx: "
                      f"{[round(float(s['nx']), 3) for s in stale]})")
            samples.append({"nx": new_nx, "ny": round(pt[1], 4),
                            "arm": [round(v, 1) for v in arm.arm]})
            samples.sort(key=lambda s: float(s["nx"]))
            print(f"[cal] sample added to row (curve ny here "
                  f"{row_ny_at(row, pt[0]):.3f}): nx={new_nx} ny={pt[1]:.4f} "
                  f"arm={[round(v, 1) for v in arm.arm]}")
            save_and_reload()
        elif line == "g":
            if not camera:
                print("no camera.")
                continue
            pt = camera.pick_point(solver.rows_for(pose),
                                   f"GRAB TEST ({pose})", pose=pose)
            if pt is None:
                print("cancelled.")
                continue
            # Click-based: a lying grab test assumes the tin points STRAIGHT at
            # the robot (angle 90). Angle-driven CH5 needs test_arc_live.py.
            solved = solver.solve(pt[0], pt[1], pose=pose)
            if solved is None:
                print(f"[solve] ({pt[0]:.3f},{pt[1]:.3f}) NOT grabbable in {pose} "
                      f"(outside band / CH1 range / not calibrated). {solver.status()}")
                continue
            if pose == "lying":
                print("[grab] lying test assumes tin points AT the robot "
                      f"(CH5={solved[4]:.0f}); angled tins: test_arc_live.py")
            if wheels is not None:
                wheels.brake()
            try:
                arm.collect(solved, tin_pose=pose, dump=False)   # ends holding
            finally:
                if wheels is not None:
                    wheels.stop()
        elif handle_servo_command(arm, line):
            pass
        else:
            print("unknown. commands: c2 145 | c1 +2 | 5 angles | p o c h pose st | "
                  "speed | grip | cam y a g | lie stand | r | q")

    if wheels is not None:
        wheels.stop()             # never leave the base braked after exit
    arm.release()                 # and never leave servos holding a pose
    if camera:
        camera.stop()
    print("bye")


if __name__ == "__main__":
    main()
