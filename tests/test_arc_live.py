"""
tests/test_arc_live.py

LIVE arc-grasp checker: YOLO detects the tin, its ground-contact pixel is fed
to ArcGraspSolver every frame, and the overlay tells you whether the tin is
GRABBABLE right now (inside a calibrated arc band) — plus which way to move
if it isn't. Modeled on test_ibvs_centering's live loop.

Meant for the "one arc calibrated, does it work?" stage: calibrate with
tests/test_arc_grasp.py first (even a single row of 1-3 samples), then run
this and slide the tin around the floor.

Usage (Pi desktop):
    python tests/test_arc_live.py

Keys:
    g = grab NOW using the solved pose (needs arm hardware; ends HOLDING
        so you can check the grip — the autonomous stack dumps by itself)
    b = dump the held tin into the onboard bin
    h = arm home (forced)
    q = quit

Overlay:
    yellow line(s)   calibrated arc(s), diamonds = azimuth samples
    green banner     GRABBABLE + the CH1-5 the solver would send
    orange banner    not grabbable + a hint (too far / too close / off side)
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.arc_grasp import (
    ArcGraspSolver, row_ny_at,
    NY_TOL_DEFAULT, NY_TOL_NEAR_DEFAULT, NX_TOL_DEFAULT,
)

IMGSZ = 640
INFER_EVERY = 1


def _make_arm():
    """Reuse test_arc_grasp's tuned grab sequence (loaded by file path so a
    missing PCA9685 only disables 'g', it doesn't kill the viewer)."""
    try:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_arc_grasp.py")
        spec = importlib.util.spec_from_file_location("arc_tool", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # raises if hardware libs missing
        arm = mod.Arm()
        print("[arm] ready — 'g' grabs, 'h' homes.")
        return arm
    except Exception as exc:
        print(f"[arm] not available ({exc}); viewer only, 'g' just prints.")
        return None


def _pose_of(box):
    """(pose, angle_deg_or_None, label) from a detection's segmentation mask.
    No usable mask -> assume upright (the historical behaviour)."""
    o = box.orientation
    if o is None or o.klass == "upright":
        return "upright", None, "upright"
    if o.klass == "axial":
        return "lying", None, "lying end-on"
    return "lying", o.angle, f"lying {o.angle:.0f}deg"


def _classify(solver, nx, ny, pose, angle_deg):
    """(solved_or_None, hint_text). Mirrors the solver's band rules so the
    hint always agrees with what solve() decided. Rows are CURVES, so the
    far/close comparison happens against each arc's height AT THIS nx."""
    solved = solver.solve(nx, ny, pose=pose, angle_deg=angle_deg)
    if solved is not None:
        return solved, ""
    if not solver.ready_for(pose):
        return None, f"{pose} arc NOT calibrated - test_arc_grasp.py " \
                     f"({'lie' if pose == 'lying' else 'stand'} mode)"
    rs = solver.rows_for(pose)
    heights = sorted((row_ny_at(r, nx), r) for r in rs)
    lo = heights[0][0] - float(heights[0][1].get("ny_tol", NY_TOL_DEFAULT))
    hi = heights[-1][0] + float(heights[-1][1].get("ny_tol_near",
                                                   NY_TOL_NEAR_DEFAULT))
    if ny < lo:
        return None, "tin TOO FAR - drive forward onto the arc"
    if ny > hi:
        return None, "tin TOO CLOSE - back up onto the arc"
    return None, "tin OFF THE ARC SIDEWAYS - outside the sampled span"


def _draw_arcs(frame, solver):
    """Upright grid in cyan, lying grid in magenta. Rows are CURVES drawn as
    polylines through the samples' own (nx, ny) points. NO filled band —
    it over-promised: the physically valid spot is ON the line or a little
    ABOVE it (farther), so we draw the bright arc plus one faint line at
    the far tolerance edge. Place the tin between those two lines."""
    import cv2
    import numpy as np
    fh, fw = frame.shape[:2]
    for pose, color in (("upright", (0, 220, 220)), ("lying", (220, 0, 220))):
        faint = tuple(int(c * 0.45) for c in color)
        for r in solver.rows_for(pose):
            default_ny = float(r["ny"])
            far_tol = float(r.get("ny_tol", NY_TOL_DEFAULT))
            xtol = float(r.get("nx_tol", NX_TOL_DEFAULT))
            pts = sorted((float(s["nx"]), float(s.get("ny", default_ny)))
                         for s in r["samples"])
            # extend flat by nx_tol beyond the end samples (solver clamps there)
            ext = ([(max(0.0, pts[0][0] - xtol), pts[0][1])] + pts +
                   [(min(1.0, pts[-1][0] + xtol), pts[-1][1])])
            px = np.array([[int(x * fw), int(y * fh)] for x, y in ext], np.int32)
            # faint upper line = how far ABOVE the arc still grabs
            cv2.polylines(frame, [px + [0, -int(far_tol * fh)]], False, faint, 1)
            # the arc itself — put the tin ON this line or slightly above
            cv2.polylines(frame, [px], False, color, 2)
            for x, y in pts:
                cv2.drawMarker(frame, (int(x * fw), int(y * fh)), color,
                               cv2.MARKER_DIAMOND, 14, 2)


def main():
    import cv2
    from src.perception.detector import AluminiumCanDetector

    solver = ArcGraspSolver()
    print(f"[cfg] {solver.status()}")
    if not solver.ready:
        print("[cfg] no calibration — the viewer runs, but nothing will be grabbable.")

    # Defaults mirror the runtime: RUNTIME_MODEL_PATH, conf 0.5, device auto
    # (NCNN export picked up automatically if present).
    detector = AluminiumCanDetector(imgsz=IMGSZ)
    detector.start()
    arm = _make_arm()

    print("\nKeys: g=grab (ends holding)  b=dump to bin  h=home  q=quit\n")
    show = True
    result = None
    solved = None
    pose = "upright"          # updated per detection; used by 'g' for the order
    i = 0

    try:
        while True:
            frame = detector.read_frame()
            if frame is None:
                continue

            best = None
            hint = ""
            if i % INFER_EVERY == 0:
                result = detector.infer(frame)
            i += 1
            best = result.best if result is not None else None

            pose_label = ""
            if best is not None:
                pose, angle, pose_label = _pose_of(best)
                # Reference point matches the calibration convention:
                # upright -> ground contact (bbox bottom-center);
                # lying   -> bbox CENTER (tracks the graspable middle at
                #            every orientation; the bottom edge doesn't).
                u, v = best.base_center if pose == "upright" else best.center
                nx = u / result.frame_width
                ny = v / result.frame_height
                solved, hint = _classify(solver, nx, ny, pose, angle)
                tag = (f"GRABBABLE {solved}" if solved is not None else hint)
                print(f"ref=({u:4d},{v:4d}) n=({nx:.3f},{ny:.3f}) [{pose_label}]  {tag}")
            else:
                solved = None
                print("[no detection]")

            if not show:
                continue

            annotated = detector.get_annotated_frame(result) if result is not None else frame
            if annotated is None:
                annotated = frame
            _draw_arcs(annotated, solver)

            fh, fw = annotated.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            if best is None:
                msg, color = "no tin detected", (200, 200, 200)
            elif solved is not None:
                msg, color = f"GRABBABLE [{pose_label}]  (press 'g')", (0, 255, 0)
            else:
                msg, color = f"[{pose_label}] {hint}", (0, 165, 255)
            cv2.putText(annotated, msg, (20, fh - 25), font, 0.9, (0, 0, 0), 5)
            cv2.putText(annotated, msg, (20, fh - 25), font, 0.9, color, 2)
            if solved is not None:
                pose_txt = "CH1-5: " + "  ".join(f"{v:.1f}" for v in solved)
                cv2.putText(annotated, pose_txt, (20, fh - 60), font, 0.6, (0, 0, 0), 4)
                cv2.putText(annotated, pose_txt, (20, fh - 60), font, 0.6, (0, 255, 0), 1)

            try:
                cv2.imshow("arc grasp live  (g=grab  h=home  r=release  q=quit)", annotated)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print("[!] no display — continuing text-only")
                show = False
                continue

            if key == ord("q"):
                break
            if key == ord("g"):
                if solved is None:
                    print("[grab] not grabbable right now.")
                elif arm is None:
                    print(f"[grab] (no hardware) would send {solved}")
                else:
                    # ends HOLDING — press 'b' to dump; the approach order
                    # follows the detected tin pose (lying: elbow last)
                    arm.grab(solved, tin_pose=pose)
            if key == ord("b"):
                if arm is None:
                    print("[bin] (no hardware) would dump to bin")
                else:
                    arm.dump_to_bin()
            if key == ord("h") and arm is not None:
                arm.force_home()
            if key == ord("r") and arm is not None:
                arm.release()   # cut PWM — arm goes limp (servo_jog-style)

    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        try:
            detector.stop()
        except Exception:
            pass
        try:
            if arm is not None:
                arm.release()   # never leave servos holding a pose after exit
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
