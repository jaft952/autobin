"""
tests/test_pixel_grasp.py

Calibrate the pixel -> arm-frame ground transform, then IK-grab a detected
tin directly — no IBVS sweet-spot needed once the tin is inside the arm's
workspace.

How it works: the tin stands on the floor, so the detection's ground-contact
pixel (bbox bottom-center) maps to floor (x, y) metres by one homography
(src/arm/pixel_to_arm.py). Grasp target = that (x, y) at tin-waist height;
GraspPlanner adds the sag compensation.

CALIBRATION (once, on the Pi desktop — needs the camera view):
  1. python tests/test_pixel_grasp.py
  2. Put the tin on a ruler-measured floor spot. Measure x (right +) and
     y (forward) in cm FROM THE POINT ON THE FLOOR DIRECTLY UNDER THE CH1
     YAW AXIS (drop a plumb line / ruler from the arm base).
  3. Press 'c' in the window, type "x y" in cm in the terminal.
  4. Repeat for >= 4 spots — spread them (near/far/left/right), e.g.
     (0,25) (0,32) (0,40) (-12,30) (12,30) (-8,38).  More = better.
  5. Press 'f' -> fits, prints per-point error in cm, saves into
     src/visual_servoing/config/centering_config.yaml (pixel_to_arm: key).

AFTER CALIBRATION the overlay shows the live predicted (x, y) for the tin.
Sanity-check with the ruler before grabbing.

Keys:
  c = capture calibration sample     u = undo last sample
  f = fit + save                     g = grab the detected tin (IK)
  h = arm home                       q = quit

If the camera is ever remounted/bumped -> recalibrate (the mapping is the
camera-to-chassis geometry).
"""
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.pixel_to_arm import PixelToArm, MIN_SAMPLES

# ── grasp geometry ───────────────────────────────────────────────────────────
# Standing 330ml tin: ~11.5cm tall, waist ~5.7cm above the floor. The floor is
# 11.3cm below the deck (DECK_ABOVE_FLOOR_M), so in IK deck-frame coords:
TIN_WAIST_Z = -0.056
# LEVEL (horizontal) grabs need the 18.65cm gripper segment between wrist and
# can -> only solvable from ~32cm out; closer tins use the DOWN (tilt) ladder.
LEVEL_MIN_REACH = 0.32

INFER_EVERY = 1      # YOLO every Nth frame (raise if the Pi CPU chokes)
IMGSZ = 640


def _make_planner():
    """GraspPlanner needs the PCA9685; keep the tool usable without hardware
    (calibration / transform-check only)."""
    try:
        from src.arm.grasp_planner import GraspPlanner
        planner = GraspPlanner()
        print("[arm] GraspPlanner ready — 'g' will really grab.")
        return planner
    except Exception as exc:
        print(f"[arm] no arm hardware ({exc}); transform-only mode, 'g' just prints.")
        return None


def _grab(planner, x, y):
    """Open -> IK move to the tin waist -> close -> carry to bin -> home."""
    from src.arm.kinematics import GRIPPER_DOWN, GRIPPER_LEVEL
    reach = math.hypot(x, y)
    tool = GRIPPER_LEVEL if reach >= LEVEL_MIN_REACH else GRIPPER_DOWN
    mode = "LEVEL" if tool is GRIPPER_LEVEL else "DOWN"
    print(f"[grab] tin at ({x:+.3f}, {y:+.3f}) m, reach {reach:.3f} -> {mode} grasp "
          f"at z={TIN_WAIST_Z}")
    if planner is None:
        print("[grab] (no hardware — would move_to and close here)")
        return
    planner.control_gripper("open")
    if not planner.move_to([x, y, TIN_WAIST_Z], tool_direction=tool):
        print(f"[grab] unreachable in {mode} mode — drive closer/farther and retry.")
        return
    planner.control_gripper("close")
    planner.dump_to_bin()    # carry to the chassis bin and release
    planner.home()
    print("[grab] done.")


def main():
    import cv2
    from src.perception.detector import AluminiumCanDetector

    p2a = PixelToArm()
    print(p2a.status())

    detector = AluminiumCanDetector(device="cpu", imgsz=IMGSZ)
    detector.start()
    planner = _make_planner()

    if not p2a.check_resolution(detector._frame_width, detector._frame_height):
        print(f"[!] calibration was captured at {p2a.cfg.get('resolution')} but the "
              f"camera is {detector._frame_width}x{detector._frame_height} — RECALIBRATE.")

    print("\nKeys: c=capture  u=undo  f=fit+save  g=grab  h=home  q=quit")
    win = "pixel grasp  (c=capture u=undo f=fit g=grab h=home q=quit)"
    result = None
    i = 0

    try:
        while True:
            frame = detector.read_frame()
            if frame is None:
                continue
            if i % INFER_EVERY == 0:
                result = detector.infer(frame)
            i += 1

            annotated = detector.get_annotated_frame(result) if result is not None else frame
            best = result.best if result is not None else None

            # Live overlay: predicted arm-frame position of the tin.
            if best is not None and p2a.ready:
                u, v = best.base_center
                pred = p2a.transform(u, v)
                if pred is not None:
                    px, py = pred
                    reach = math.hypot(px, py)
                    mode = "LEVEL" if reach >= LEVEL_MIN_REACH else "DOWN"
                    cv2.putText(annotated,
                                f"arm frame: x={px * 100:+.1f}cm y={py * 100:+.1f}cm "
                                f"reach={reach * 100:.1f}cm [{mode}]",
                                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                    cv2.putText(annotated,
                                f"arm frame: x={px * 100:+.1f}cm y={py * 100:+.1f}cm "
                                f"reach={reach * 100:.1f}cm [{mode}]",
                                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
            elif not p2a.ready:
                cv2.putText(annotated,
                            f"NOT calibrated: {len(p2a.samples)}/{MIN_SAMPLES} samples "
                            f"(place tin, press 'c')",
                            (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

            cv2.imshow(win, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                if best is None:
                    print("[cal] no tin detected — place the tin in view first.")
                    continue
                u, v = best.base_center
                print(f"[cal] captured pixel ({u}, {v}).")
                try:
                    raw = input("      real position 'x y' in cm (floor, from under CH1 axis): ")
                    x_cm, y_cm = (float(t) for t in raw.strip().split())
                except (ValueError, EOFError):
                    print("      bad input — sample discarded.")
                    continue
                p2a.add_sample(u, v, x_cm / 100.0, y_cm / 100.0)
                print(f"      sample #{len(p2a.samples)}: px({u},{v}) -> "
                      f"({x_cm:.1f}, {y_cm:.1f}) cm")

            elif key == ord('u'):
                gone = p2a.undo_sample()
                print(f"[cal] removed {gone}" if gone else "[cal] nothing to undo.")

            elif key == ord('f'):
                res = p2a.fit(detector._frame_width, detector._frame_height)
                if res is None:
                    print(f"[cal] need >= {MIN_SAMPLES} samples (have {len(p2a.samples)}).")
                    continue
                p2a.save()
                errs = ", ".join(f"{e:.1f}" for e in res)
                print(f"[cal] fitted + saved. per-sample error: [{errs}] cm "
                      f"(mean {sum(res) / len(res):.1f} cm)")
                if max(res) > 2.0:
                    print("[cal] a point is off by >2cm — re-measure it or add more samples.")

            elif key == ord('g'):
                if not p2a.ready:
                    print("[grab] calibrate first ('c' x4 then 'f').")
                elif best is None:
                    print("[grab] no tin detected.")
                else:
                    u, v = best.base_center
                    pred = p2a.transform(u, v)
                    if pred is None:
                        print("[grab] transform failed (degenerate calibration?).")
                    else:
                        _grab(planner, pred[0], pred[1])

            elif key == ord('h') and planner is not None:
                planner.home()

    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        try:
            detector.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
