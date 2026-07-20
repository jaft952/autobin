"""
Real-time Visual Servoing with Cascade Control (multi-rate, smooth motion).

Usage: python test_ibvs_centering.py [--drive] [--arm] [--speed 0.x] [--turn 0.x]
                                     [--mode step|chase] [--pt]
  --drive : enable chassis motion (default off)
  --arm   : enable arc-grasp grabbing (default off). Checked every frame on
            the RAW detection (same pipeline as tests/test_arc_live.py) — the
            top-left readout shows GRABBABLE (green) or the reason it isn't
            (TOO FAR / TOO CLOSE / OFF SIDEWAYS, orange), and the cyan/magenta
            arc overlay shows the calibrated grab band on the live feed.
  --speed : forward speed multiplier, 0..1 (default 1.0, too fast? lower it).
            MAX-MIN form e.g. --speed 0.6-0.3: far from the tin runs at 0.6,
            tapers down as it approaches but never below 0.3 (no stall crawl)
  --turn  : steer speed multiplier, 0..1 (default: same as --speed; turning has
            no rolling friction, so it usually wants a LOWER value, e.g.
            --speed 0.6 --turn 0.3)
  --mode  : step   = aim, drive straight, re-aim when tin leaves central 70% (default)
            chase  = car-like continuous pursuit: drive + steer at the same time,
                     pivoting only if the tin nears the frame edge
            cruise = METRIC approach: projects the tin onto the floor (metres),
                     plans a drive to the grasp spot, and dead-reckons the
                     remaining distance from the wheel duties between frames,
                     so it decelerates on schedule instead of executing a
                     stale full-speed command until the next detection lands
                     (that lag is what made it push the tin). Final 12 cm are
                     small pulses with a pause between for a fresh look.
                     Needs the floor calibration; without it, falls back to
                     the old pixel chase. Watch the yellow PLAN d=... readout.
            (all brake inside the arc-grasp zone / on lost detection)
  --pt    : force the .pt weights, skip NCNN (A/B comparison)
Keys: m = toggle drive, g = toggle arm, n = cycle step/chase/cruise, q = quit.
A grab brakes the base and runs the blocking sequence from tests/test_arc_grasp.py
(the SAME Arm.grab you tune with 'g' there) -> dump into the bin -> home, with a
4 s cooldown, then releases the brake.
"""

import os
import sys
import time
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # for test_arc_grasp

from src.visual_servoing.ibvs_centering import IBVSCentering
from src.visual_servoing.cascade_controller import CascadeController
# arc_grasp has no heavy deps (no torch/ikpy) — safe to import at module
# level, unlike GraspPlanner. Ported from tests/test_arc_live.py so the arc
# overlay / hint text / grab decision all use the exact same solver rules.
from src.arm.arc_grasp import (
    row_ny_at, NY_TOL_DEFAULT, NY_TOL_NEAR_DEFAULT, NX_TOL_DEFAULT,
)

# Same cooldown ArmExecutor used: a tin still visible mid-lift (or a missed
# grab) must not re-trigger the whole sequence on the very next loop.
GRAB_COOLDOWN_S = 4.0


def _interpret_motion_command(cmd):
    """Human-readable motion from a MotorCommand.

    Sign convention (hardware-verified via test_differential_drive CASC):
    positive forward = robot forward, positive steer = turn LEFT."""
    if cmd is None:
        return "IDLE", 0, 0

    forward = cmd.forward
    steer = cmd.steer
    mag = math.sqrt(forward**2 + steer**2)

    if mag < 0.05:
        return "HOLD", 0, 0

    side = "LEFT" if steer > 0 else "RIGHT"
    if abs(steer) < 0.1:
        motion = "FORWARD" if forward > 0 else "BACKWARD"
        intensity = abs(forward)
    elif abs(forward) < 0.1:
        motion = f"TURN {side}"
        intensity = abs(steer)
    else:
        motion = f"ARC {'FWD' if forward > 0 else 'BACK'}-{side}"
        intensity = mag

    return motion, steer, intensity


def _pose_of(box):
    """(pose, angle_deg_or_None, label) from a detection's segmentation mask.
    No usable mask -> assume upright. Ported verbatim from
    tests/test_arc_live.py so both tools agree on what a detection IS."""
    o = box.orientation
    if o is None or o.klass == "upright":
        return "upright", None, "upright"
    if o.klass == "axial":
        return "lying", None, "lying end-on"
    return "lying", o.angle, f"lying {o.angle:.0f}deg"


def _classify(solver, nx, ny, pose, angle_deg):
    """(solved_or_None, hint_text). Mirrors the solver's band rules so the
    hint always agrees with what solve() decided. Ported from
    tests/test_arc_live.py — the TOO FAR/TOO CLOSE/SIDEWAYS breakdown that
    used to just be a flat 'outside calibrated grid'."""
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
    """Upright grid in cyan, lying grid in magenta — the calibrated grab
    band drawn straight onto the live feed, so you can SEE the zone and
    watch the tin approach it instead of reading numbers off the console.
    Ported from tests/test_arc_live.py."""
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
            ext = ([(max(0.0, pts[0][0] - xtol), pts[0][1])] + pts +
                   [(min(1.0, pts[-1][0] + xtol), pts[-1][1])])
            px = np.array([[int(x * fw), int(y * fh)] for x, y in ext], np.int32)
            cv2.polylines(frame, [px + [0, -int(far_tol * fh)]], False, faint, 1)
            cv2.polylines(frame, [px], False, color, 2)
            for x, y in pts:
                cv2.drawMarker(frame, (int(x * fw), int(y * fh)), color,
                               cv2.MARKER_DIAMOND, 14, 2)


def _grab_readout(solver, result):
    """(pose, pose_label, pos, solved, hint) for the CURRENT raw (unsmoothed)
    detection — the exact pipeline _attempt_grab uses to decide whether to
    grab, shared here so the on-screen text can never disagree with the
    actual decision. pos is None if there's nothing to classify."""
    if solver is None or result is None or result.best is None:
        return "upright", "", None, None, ""
    pose, angle, pose_label = _pose_of(result.best)
    pos = (result.normalized_center() if pose == "lying"
           else result.normalized_base_center())
    if pos is None:
        return pose, pose_label, None, None, ""
    solved, hint = _classify(solver, pos[0], pos[1], pose, angle)
    return pose, pose_label, pos, solved, hint


def _plan_text(controller):
    """One-line summary of the metric approach plan, or '' in pixel mode."""
    planner = getattr(controller, "planner", None)
    if planner is None or not planner.has_plan:
        return ""
    cmd = controller.get_last_command()
    phase = cmd.phase if cmd is not None and cmd.phase else "-"
    return (f"PLAN {phase:5} d={planner.distance():.2f}m "
            f"hdg={math.degrees(planner.heading()):+.0f}deg")


def _draw_status_overlay(frame, status, cmd, driving=None, sent=(0.0, 0.0), plan="",
                         grab_text=("", False)):
    """Draw cascade status and motion command on frame."""
    import cv2
    if frame is None:
        return
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    banner_top = fh - 110  # dark banner so text stays legible over a busy background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, banner_top), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Big direction indicator (bottom-left): which way it is about to move.
    motion, steer, intensity = _interpret_motion_command(cmd)
    if intensity > 0:
        arrow = "<<< " if steer > 0.05 else (" >>>" if steer < -0.05 else "")
        big = f"{arrow}{motion}{arrow}" if arrow else motion
        cv2.putText(frame, big, (20, fh - 72), font, 1.1, (0, 255, 255), 3)
    else:
        cv2.putText(frame, motion, (20, fh - 72), font, 1.1, (170, 170, 170), 3)

    motion_color = (0, 255, 0) if intensity > 0 else (170, 170, 170)
    motion_text = (f"Motion: {motion}  |  speed fwd={abs(sent[0]):.2f} "
                   f"turn={abs(sent[1]):.2f}")
    cv2.putText(frame, motion_text, (20, fh - 42), font, 0.7, motion_color, 2)

    # Metric plan (cruise): how far the planner still thinks it has to go.
    # Watch this while tuning — if it hits 0.00 while the tin is still ahead,
    # the learned v_max is too high; if it stops short, too low.
    if plan:
        cv2.putText(frame, plan, (fw - 430, 64), font, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, plan, (fw - 430, 64), font, 0.65, (255, 220, 0), 1)

    # Alignment / stability (lower line)
    color = (0, 255, 0) if status.stable else (0, 165, 255)
    text = f"aligned={status.aligned}  stable={status.stable}  quality={status.quality():.2f}"
    cv2.putText(frame, text, (20, fh - 14), font, 0.7, color, 2)

    if driving is not None:
        dmode = "DRIVE ON" if driving else "DRIVE OFF"
        dcol = (0, 255, 0) if driving else (255, 0, 0)
        cv2.putText(frame, dmode, (fw - 250, 32), font, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, dmode, (fw - 250, 32), font, 0.7, dcol, 1)

    # Arm grab readout (top-left): GRABBABLE in green, or the specific
    # TOO FAR/TOO CLOSE/SIDEWAYS reason in orange — the exact same
    # _grab_readout pipeline _attempt_grab uses, so this can never show
    # "ready" while the arm actually refuses (or vice versa).
    msg, ok = grab_text
    if msg:
        gcol = (0, 255, 0) if ok else (0, 165, 255)
        cv2.putText(frame, msg, (20, 32), font, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, msg, (20, 32), font, 0.65, gcol, 1)


def _make_arm(chassis=None):
    """Init arc-grasp solver + the SAME Arm class tests/test_arc_grasp.py uses.

    Driving the calibration tool's Arm directly (instead of ArmExecutor ->
    GraspPlanner) means the grab that runs here is byte-for-byte the one you
    tune with 'g' in that tool: same lift pose, same per-tin-pose channel
    order, same gripper limits, same persisted pose file."""
    try:
        from src.arm.arc_grasp import ArcGraspSolver
        from test_arc_grasp import Arm
        solver = ArcGraspSolver()
        if not solver.ready: # type: ignore
            print(f"[arm] arc_grasp calibration not ready ({solver.status()}) — arm mode unavailable.")
            return None, None
        # ONE motor driver per process: hand the Arm the chassis' actuator
        # rather than letting it open a second one — PWMActuator.__init__
        # runs GPIO.cleanup on the shared motor pins and would kill the
        # chassis' PWM channels mid-run.
        arm = Arm(wheels=chassis.actuator) if chassis is not None else Arm()
        print("[arm] arc-grasp ready (test_arc_grasp.Arm) — press 'g' to toggle grabbing.")
        return arm, solver
    except Exception as exc:
        print(f"[arm] not available ({exc}); arm mode disabled.")
        return None, None


def _attempt_grab(controller, chassis, solver, arm, result, state):
    """Solve an arc pose from the CURRENT (raw, per-frame) detection and run
    test_arc_grasp's blocking grab -> dump -> home.

    Checked every frame via _grab_readout — the SAME raw-detection pipeline
    tests/test_arc_live.py uses — rather than gating on IBVS's debounced
    'stable' flag. The arc solver's own nx_tol/ny_tol margins are already the
    real test of "close enough to grab"; a second, independently-tuned
    debounce on top of that only delayed grabs the arm could already make
    (and, before this pass, could disagree with it entirely — see
    ibvs_centering.py's arc-authoritative aligned check).

    Brakes the base for the whole sequence, then restores the drive state."""
    if time.monotonic() < state["cooldown_until"]:
        return

    pose, pose_label, pos, solved, hint = _grab_readout(solver, result)
    if pos is None:
        return
    if solved is None:
        print(f"[arm] {pose_label} tin at nx={pos[0]:.2f} ny={pos[1]:.2f} — {hint}")
        # Cooldown on failure too: a tin sitting just outside the grid stays
        # there for many frames, so without this it would re-solve and
        # re-print every single loop instead of once per cooldown window.
        state["cooldown_until"] = time.monotonic() + GRAB_COOLDOWN_S
        return

    print(f"[arm] GRAB ({pose_label}) @ nx={pos[0]:.2f} ny={pos[1]:.2f} — braking base...")
    was_driving = controller.driving
    controller.set_driving(False)
    try:
        if not state["homed"]:
            # First arm move of the session: the tracked pose came from the
            # last session's file, so assert home before trusting it.
            arm.force_home()
            state["homed"] = True
        arm.grab(solved, tin_pose=pose)        # brakes the wheels internally
        arm.brake_wheels()                    # grab() released it — hold for the carry
        try:
            arm.dump_to_bin()                 # bin pose + open
            arm.force_home()
        finally:
            arm.release_wheels()
    finally:
        state["cooldown_until"] = time.monotonic() + GRAB_COOLDOWN_S
        if chassis is not None:
            chassis.stop()  # release brake -> coast
        controller.set_driving(was_driving)
    print("[arm] grab sequence done, drive restored.")


def _make_chassis():
    """Initialize chassis controller (optional)."""
    try:
        from src.visual_servoing.chassis_controller import ChassisController
        chassis = ChassisController()
        print("[drive] chassis ready — press 'm' to toggle motion on/off.")
        return chassis
    except Exception as exc:
        print(f"[drive] chassis not available ({exc}); motion disabled.")
        return None


def main(drive=False, speed=1.0, use_ncnn=True, arm=False, turn=None, mode="step",
         speed_min=0.0):
    """Real-time visual servoing with cascade control (multi-rate, smooth motion)."""
    import cv2
    from src.perception.detector import AluminiumCanDetector, RUNTIME_MODEL_PATH

    IMGSZ = 640

    # 720p (matches CameraSensor runtime): the 1080p default overflowed the
    # VNC screen, hiding the bottom status banner entirely.
    detector = AluminiumCanDetector(  # type: ignore
        device="cpu", imgsz=IMGSZ, model_path=RUNTIME_MODEL_PATH, use_ncnn=use_ncnn,
        frame_width=1280, frame_height=720,
    )
    detector.start()
    centering = IBVSCentering()
    chassis = _make_chassis()
    driving = drive and chassis is not None

    arm_ctl, solver = _make_arm(chassis) if arm else (None, None)
    armed = arm and arm_ctl is not None
    grab_state = {"cooldown_until": 0.0, "homed": False}

    print(f"\n{'='*70}")
    print(f"Visual Servoing + Cascade Control")
    print(f"{'='*70}")
    print(f"Vision: ~30 Hz (YOLO11n-seg + IBVS)")
    print(f"Motor: ~1000 Hz (smooth interpolation)")
    print(f"Drive: {'ON' if driving else 'OFF'}   Arm: {'ARMED' if armed else 'OFF'}   Mode: {mode.upper()}")
    spd_txt = f"{speed:.2f}" if speed_min <= 0 else f"max={speed:.2f} min={speed_min:.2f}"
    print(f"Speed scale: fwd={spd_txt} turn={(turn if turn is not None else speed):.2f}  "
          f"(--speed / --turn 0.x to slow down)")
    print(f"Keys: m = toggle drive, g = toggle arm, q = quit")
    print(f"{'='*70}\n")

    controller = CascadeController(detector, centering, chassis, speed_scale=speed,
                                   steer_scale=turn, driving=driving, mode=mode,
                                   speed_min=speed_min)
    controller.start()

    show = True
    status = None

    try:
        while True:
            # Get latest vision status and motor command
            status = controller.get_status()
            cmd = controller.get_last_command()

            # Draw from the vision thread's cached result — avoids a second read_frame() race.
            if show:
                try:
                    frame = None
                    if status is not None and status.result is not None:
                        frame = detector.get_annotated_frame(status.result) # type: ignore

                    if frame is not None:
                        if solver is not None:
                            _draw_arcs(frame, solver)
                        _, _, _, solved, hint = _grab_readout(solver, status.result)  # type: ignore
                        grab_text = ("GRABBABLE (arm will grab)" if solved is not None
                                    else hint, solved is not None)
                        _draw_status_overlay(frame, status, cmd,
                                             driving if chassis is not None else None,
                                             sent=controller.get_last_sent(),
                                             plan=_plan_text(controller),
                                             grab_text=grab_text)
                    else:
                        import numpy as np
                        frame = np.zeros((540, 960, 3), dtype=np.uint8)
                        cv2.putText(frame, "[Waiting for first camera frame + detection...]",
                                    (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                    cv2.imshow("Visual Servoing + Cascade  (m=drive, q=quit)", frame)  # type: ignore
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("m") and chassis is not None:
                        driving = not driving
                        controller.set_driving(driving)  # actually gate the wheels
                        print(f"\n[mode] drive toggled: {'ON' if driving else 'OFF'}\n")
                    if key == ord("g"):
                        if arm_ctl is None:
                            arm_ctl, solver = _make_arm(chassis)
                        if arm_ctl is not None:
                            armed = not armed
                            print(f"\n[mode] arm {'ARMED — grabs when centered+stable' if armed else 'off'}\n")
                    if key == ord("n"):
                        mode = {"step": "chase", "chase": "cruise", "cruise": "step"}[mode]
                        controller.set_mode(mode)
                        print(f"\n[mode] pursuit mode -> {mode.upper()}\n")

                except cv2.error:
                    print("[!] no display available — continuing text-only")
                    show = False
                except Exception as e:
                    print(f"[display] error: {e}")

            # Grab when armed, checked on the RAW per-frame detection every
            # tick (mirrors tests/test_arc_live.py) rather than gated behind
            # IBVS's debounced 'stable' flag — the arc solver's own nx_tol/
            # ny_tol margins are already the real "close enough" test.
            # GRAB_COOLDOWN_S in grab_state stops the same tin re-triggering
            # the sequence every loop once a decision (grab or refuse) is made.
            if armed and arm_ctl is not None and status is not None and status.result is not None:
                _attempt_grab(controller, chassis, solver, arm_ctl,
                              status.result, grab_state)

            # Print status to console with motion details
            if status and cmd:
                motion, _, intensity = _interpret_motion_command(cmd)
                plan = _plan_text(controller)
                print(
                    f"err=({status.error_x:+.3f},{status.error_y:+.3f})  "
                    f"mask_area={status.mask_area if status.mask_area else 'None':>7}  "
                    f"quality={status.quality():.2f}  "
                    f"aligned={status.aligned}  stable={status.stable}  "
                    f"| motion={motion:20}  intensity={intensity:.2f}  "
                    f"drive={'ON' if driving else 'OFF'}"
                    + (f"  | {plan}" if plan else "")
                )
            elif status:
                print(
                    f"err=({status.error_x:+.3f},{status.error_y:+.3f})  "
                    f"mask_area={status.mask_area if status.mask_area else 'None':>7}  "
                    f"quality={status.quality():.2f}  "
                    f"aligned={status.aligned}  stable={status.stable}  "
                    f"drive={'ON' if driving else 'OFF'}"
                )
            else:
                print("[waiting for first detection...]")

            time.sleep(0.05)  # Monitor at ~20Hz (don't spam console)

    except KeyboardInterrupt:
        print("\n[stopped by user]")
    finally:
        try:
            controller.stop()
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
            cv2.destroyAllWindows()  # type: ignore
        except Exception:
            pass
        print("\n[cleanup complete]")


if __name__ == "__main__":
    drive = "--drive" in sys.argv
    arm = "--arm" in sys.argv
    speed, speed_min = 1.0, 0.0
    if "--speed" in sys.argv:
        raw = sys.argv[sys.argv.index("--speed") + 1]
        parts = raw.split("-")
        speed = float(parts[0])
        if len(parts) > 1:
            speed_min = float(parts[1])
            if speed_min > speed:
                sys.exit(f"--speed MAX-MIN: max ({speed}) must be >= min ({speed_min})")
    turn = None
    if "--turn" in sys.argv:
        turn = float(sys.argv[sys.argv.index("--turn") + 1])
    mode = "step"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
        if mode not in ("step", "chase", "cruise"):
            sys.exit(f"--mode must be 'step', 'chase' or 'cruise', got '{mode}'")
    use_ncnn = "--pt" not in sys.argv
    main(drive=drive, speed=speed, use_ncnn=use_ncnn, arm=arm, turn=turn, mode=mode,
         speed_min=speed_min)
