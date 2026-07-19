"""
Real-time Visual Servoing with Cascade Control (multi-rate, smooth motion).

Usage: python test_ibvs_centering.py [--drive] [--arm] [--speed 0.x] [--turn 0.x] [--pt]
  --drive : enable chassis motion (default off)
  --arm   : enable arc-grasp grabbing when centered+stable (default off)
  --speed : forward speed multiplier, 0..1 (default 1.0, too fast? lower it)
  --turn  : steer speed multiplier, 0..1 (default: same as --speed; turning has
            no rolling friction, so it usually wants a LOWER value, e.g.
            --speed 0.6 --turn 0.3)
  --pt    : force the .pt weights, skip NCNN (A/B comparison)
Keys: m = toggle drive, g = toggle arm, q = quit.
A grab brakes the base, runs the blocking arc-grasp sequence (grab -> dump ->
home, 4 s cooldown via ArmExecutor), then releases the brake.
"""

import os
import sys
import time
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visual_servoing.ibvs_centering import IBVSCentering
from src.visual_servoing.cascade_controller import CascadeController


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


def _draw_status_overlay(frame, status, cmd, driving=None):
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
    motion_text = f"Motion: {motion}  |  intensity={intensity:.2f}"
    cv2.putText(frame, motion_text, (20, fh - 42), font, 0.7, motion_color, 2)

    # Alignment / stability (lower line)
    color = (0, 255, 0) if status.stable else (0, 165, 255)
    text = f"aligned={status.aligned}  stable={status.stable}  quality={status.quality():.2f}"
    cv2.putText(frame, text, (20, fh - 14), font, 0.7, color, 2)

    if driving is not None:
        dmode = "DRIVE ON" if driving else "DRIVE OFF"
        dcol = (0, 255, 0) if driving else (255, 0, 0)
        cv2.putText(frame, dmode, (fw - 250, 32), font, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, dmode, (fw - 250, 32), font, 0.7, dcol, 1)


def _make_arm():
    """Init arc-grasp solver + executor (optional)."""
    try:
        from src.arm.arc_grasp import ArcGraspSolver
        from src.subsumption.arm_executor import ArmExecutor
        solver = ArcGraspSolver()
        if not solver.ready():
            print(f"[arm] arc_grasp calibration not ready ({solver.status()}) — arm mode unavailable.")
            return None, None
        executor = ArmExecutor()
        print("[arm] arc-grasp ready — press 'g' to toggle grabbing.")
        return executor, solver
    except Exception as exc:
        print(f"[arm] not available ({exc}); arm mode disabled.")
        return None, None


def _attempt_grab(controller, chassis, solver, arm_exec, result):
    """Solve an arc pose from the current detection and run the blocking grab.
    Brakes the base during the arm sequence, then restores the drive state."""
    from src.subsumption.arbitrator import ActionCommand

    best = result.best
    if best is None:
        return
    o = best.orientation
    klass = o.klass if o is not None else "upright"
    if klass in ("lying", "axial"):
        pos = result.normalized_center()  # same reference point as layer3
        tin_pose = klass
        solved = solver.solve(pos[0], pos[1], pose="lying",
                              angle_deg=None if klass == "axial" else o.angle) if pos else None
    else:
        pos = result.normalized_base_center()  # ground contact
        tin_pose = "upright"
        solved = solver.solve(pos[0], pos[1], pose="upright") if pos else None
    if solved is None:
        print(f"[arm] tin ({klass}) at {pos} outside calibrated grid — no grab.")
        return

    print(f"[arm] GRAB ({tin_pose}) @ nx={pos[0]:.2f} ny={pos[1]:.2f} — braking base...")
    was_driving = controller.driving
    controller.set_driving(False)
    try:
        if chassis is not None:
            chassis.actuator.brake()  # hold base against arm shake
        arm_exec.execute(ActionCommand(
            layer_id=3, active=True, motion_vector=(0, 0, 0),
            arm_action='grab_arc',
            arm_params={'pose': solved, 'tin_pose': tin_pose},
            message='ibvs test grab'))
    finally:
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


def main(drive=False, speed=1.0, use_ncnn=True, arm=False, turn=None):
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

    arm_exec, solver = _make_arm() if arm else (None, None)
    armed = arm and arm_exec is not None

    print(f"\n{'='*70}")
    print(f"Visual Servoing + Cascade Control")
    print(f"{'='*70}")
    print(f"Vision: ~30 Hz (YOLO11n-seg + IBVS)")
    print(f"Motor: ~1000 Hz (smooth interpolation)")
    print(f"Drive: {'ON' if driving else 'OFF'}   Arm: {'ARMED' if armed else 'OFF'}")
    print(f"Speed scale: fwd={speed:.2f} turn={(turn if turn is not None else speed):.2f}  "
          f"(--speed / --turn 0.x to slow down)")
    print(f"Keys: m = toggle drive, g = toggle arm, q = quit")
    print(f"{'='*70}\n")

    controller = CascadeController(detector, centering, chassis, speed_scale=speed,
                                   steer_scale=turn, driving=driving)
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
                        frame = detector.get_annotated_frame(status.result)

                    if frame is not None:
                        _draw_status_overlay(frame, status, cmd, driving if chassis is not None else None)
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
                        if arm_exec is None:
                            arm_exec, solver = _make_arm()
                        if arm_exec is not None:
                            armed = not armed
                            print(f"\n[mode] arm {'ARMED — grabs when centered+stable' if armed else 'off'}\n")

                except cv2.error:
                    print("[!] no display available — continuing text-only")
                    show = False
                except Exception as e:
                    print(f"[display] error: {e}")

            # Grab when armed and IBVS reports centered + stable (ArmExecutor's
            # own 4 s cooldown stops the same tin re-triggering every loop).
            if (armed and arm_exec is not None and status is not None
                    and status.stable and status.result is not None):
                _attempt_grab(controller, chassis, solver, arm_exec, status.result)

            # Print status to console with motion details
            if status and cmd:
                motion, _, intensity = _interpret_motion_command(cmd)
                print(
                    f"err=({status.error_x:+.3f},{status.error_y:+.3f})  "
                    f"mask_area={status.mask_area if status.mask_area else 'None':>7}  "
                    f"quality={status.quality():.2f}  "
                    f"aligned={status.aligned}  stable={status.stable}  "
                    f"| motion={motion:20}  intensity={intensity:.2f}  "
                    f"drive={'ON' if driving else 'OFF'}"
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
    speed = 1.0
    if "--speed" in sys.argv:
        speed = float(sys.argv[sys.argv.index("--speed") + 1])
    turn = None
    if "--turn" in sys.argv:
        turn = float(sys.argv[sys.argv.index("--turn") + 1])
    use_ncnn = "--pt" not in sys.argv
    main(drive=drive, speed=speed, use_ncnn=use_ncnn, arm=arm, turn=turn)
