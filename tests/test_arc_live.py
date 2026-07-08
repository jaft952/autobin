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
    g = grab NOW using the solved pose (needs arm hardware)
    h = arm home
    q = quit

Overlay:
    yellow line(s)   calibrated arc(s), diamonds = azimuth samples
    green banner     GRABBABLE + the CH1-5 the solver would send
    orange banner    not grabbable + a hint (too far / too close / off side)
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.arc_grasp import ArcGraspSolver, NY_TOL_DEFAULT, NX_TOL_DEFAULT

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
    hint always agrees with what solve() decided."""
    solved = solver.solve(nx, ny, pose=pose, angle_deg=angle_deg)
    if solved is not None:
        return solved, ""
    if not solver.ready_for(pose):
        return None, f"{pose} arc NOT calibrated - test_arc_grasp.py " \
                     f"({'lie' if pose == 'lying' else 'stand'} mode)"
    rs = solver.rows_for(pose)
    lo = float(rs[0]["ny"]) - float(rs[0].get("ny_tol", NY_TOL_DEFAULT))
    hi = float(rs[-1]["ny"]) + float(rs[-1].get("ny_tol", NY_TOL_DEFAULT))
    if ny < lo:
        return None, "tin TOO FAR - drive forward onto the arc"
    if ny > hi:
        return None, "tin TOO CLOSE - back up onto the arc"
    return None, "tin OFF THE ARC SIDEWAYS - outside the sampled span"


def _draw_arcs(frame, solver):
    """Upright grid in cyan, lying grid in magenta."""
    import cv2
    fh, fw = frame.shape[:2]
    for pose, color in (("upright", (0, 220, 220)), ("lying", (220, 0, 220))):
        for r in solver.rows_for(pose):
            ss = r["samples"]
            yy = int(float(r["ny"]) * fh)
            ntol = float(r.get("ny_tol", NY_TOL_DEFAULT))
            xtol = float(r.get("nx_tol", NX_TOL_DEFAULT))
            x1 = int(max(0.0, float(ss[0]["nx"]) - xtol) * fw)
            x2 = int(min(1.0, float(ss[-1]["nx"]) + xtol) * fw)
            # grabbable band of this row (its own ny tolerance)
            y1, y2 = int((float(r["ny"]) - ntol) * fh), int((float(r["ny"]) + ntol) * fh)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)
            # the arc itself, clipped to its sampled span
            cv2.line(frame, (x1, yy), (x2, yy), color, 2)
            for s in ss:
                cv2.drawMarker(frame, (int(float(s["nx"]) * fw), yy), color,
                               cv2.MARKER_DIAMOND, 14, 2)


def main():
    import cv2
    from src.perception.detector import AluminiumCanDetector

    solver = ArcGraspSolver()
    print(f"[cfg] {solver.status()}")
    if not solver.ready:
        print("[cfg] no calibration — the viewer runs, but nothing will be grabbable.")

    detector = AluminiumCanDetector(device="cpu", imgsz=IMGSZ)
    detector.start()
    arm = _make_arm()

    print("\nKeys: g=grab  h=home  q=quit\n")
    show = True
    result = None
    solved = None
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
                cv2.imshow("arc grasp live  (g=grab  h=home  q=quit)", annotated)
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
                    arm.grab(solved)
            if key == ord("h") and arm is not None:
                arm.set_gripper(0.0)
                arm.goto([96.7, 96.7, 150.0, 20.0, 90.0], "home")

    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        try:
            detector.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
