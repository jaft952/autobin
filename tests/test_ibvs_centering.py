"""
Real-time Visual Servoing with Cascade Control (multi-rate, smooth motion).

Architecture:
  - Vision thread: ~30 Hz (YOLO11n-seg detection + IBVS error computation)
  - Motor thread: ~1000 Hz (smooth interpolation + velocity limiting)
  - Non-blocking synchronization via TrackingBuffer
  - Dynamic filter tuning based on mask_area confidence

Usage:
    python test_ibvs_centering.py                 -> cascade control (default)
    python test_ibvs_centering.py --drive         -> cascade control + chassis motion ON
    python test_ibvs_centering.py --drive --speed 0.5   -> half speed (too fast? turn this down)
    python test_ibvs_centering.py --pt            -> force the .pt weights (skip NCNN)

--speed scales EVERY motor command sent to the chassis (0..1, default 1.0).
It's applied after cascade_controller's own MIN_SPEED floor, so even the
weakest correction gets scaled down too — lower this first if the base
lunges toward the tin too fast; the forward/steer ratio (curve shape) is
unaffected.

--pt forces the .pt checkpoint even if an NCNN export sits next to it
(default: NCNN preferred when present, auto-falls back to .pt otherwise —
see AluminiumCanDetector's use_ncnn param). Use this to A/B compare, or to
sidestep an NCNN-specific issue without deleting the export.
"""

import os
import sys
import time
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visual_servoing.ibvs_centering import IBVSCentering
from src.visual_servoing.cascade_controller import CascadeController


def _interpret_motion_command(cmd):
    """Convert motor command to human-readable motion description."""
    if cmd is None:
        return "IDLE", 0, 0

    forward = cmd.forward
    steer = cmd.steer

    # Determine motion type
    mag = math.sqrt(forward**2 + steer**2)

    if mag < 0.05:
        motion = "HOLD"
        angle = 0
        intensity = 0
    elif abs(steer) < 0.1:
        # Mostly forward/backward
        if forward < -0.2:
            motion = "FORWARD"
            angle = 0
            intensity = abs(forward)
        elif forward > 0.2:
            motion = "BACKWARD"
            angle = 0
            intensity = abs(forward)
        else:
            motion = "HOLD"
            angle = 0
            intensity = 0
    else:
        # Arc motion with turn
        angle_deg = math.atan2(steer, -forward) * 180 / math.pi  # Convert to angle

        if forward < -0.1:
            motion = f"ARC_FWD ({angle_deg:+.0f}°)"
            angle = angle_deg
            intensity = abs(forward)
        elif forward > 0.1:
            motion = f"ARC_BACK ({angle_deg:+.0f}°)"
            angle = angle_deg
            intensity = abs(forward)
        else:
            motion = f"TURN ({angle_deg:+.0f}°)"
            angle = angle_deg
            intensity = abs(steer)

    return motion, angle, intensity


def _draw_status_overlay(frame, status, cmd, driving=None):
    """Draw cascade status and motion command on frame."""
    import cv2
    if frame is None:
        return
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Solid banner behind the two status lines — plain outlined text got lost
    # against a busy background, and the two lines sat close enough (25px
    # apart at this font scale) to visually run into each other.
    banner_top = fh - 72
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, banner_top), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Motion command (upper line)
    motion, angle, intensity = _interpret_motion_command(cmd)
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


def main(drive=False, speed=1.0, use_ncnn=True):
    """Real-time visual servoing with cascade control (multi-rate, smooth motion)."""
    import cv2
    from src.perception.detector import AluminiumCanDetector, RUNTIME_MODEL_PATH

    IMGSZ = 640

    # RUNTIME_MODEL_PATH (src/models/best.pt) — same model the real runtime,
    # test_arc_grasp.py, and the NCNN export target. The old hardcoded
    # "yolov11n-seg.pt" here had no matching *_ncnn_model export, which is
    # why this test never picked up NCNN acceleration.
    detector = AluminiumCanDetector(  # type: ignore
        device="cpu", imgsz=IMGSZ, model_path=RUNTIME_MODEL_PATH, use_ncnn=use_ncnn
    )
    detector.start()
    centering = IBVSCentering()
    chassis = _make_chassis()
    driving = drive and chassis is not None

    print(f"\n{'='*70}")
    print(f"Visual Servoing + Cascade Control")
    print(f"{'='*70}")
    print(f"Vision: ~30 Hz (YOLO11n-seg + IBVS)")
    print(f"Motor: ~1000 Hz (smooth interpolation)")
    print(f"Drive: {'ON' if driving else 'OFF'}")
    print(f"Speed scale: {speed:.2f}  (--speed 0.x to slow the approach down)")
    print(f"Keys: m = toggle drive, q = quit")
    print(f"{'='*70}\n")

    # Create cascade controller (non-blocking threads). driving=driving here is
    # the REAL gate on wheel output — chassis is wired in regardless of this
    # flag, so without it the base would move on every detection even while
    # this script's own `driving` variable (used only for the on-screen label)
    # still read False.
    controller = CascadeController(detector, centering, chassis, speed_scale=speed, driving=driving)
    controller.start()

    show = True
    status = None

    try:
        while True:
            # Get latest vision status and motor command
            status = controller.get_status()
            cmd = controller.get_last_command()

            # Always try to show camera feed. Draw from the vision thread's
            # OWN cached result (status.result) via get_annotated_frame() —
            # this gives us the bbox/mask overlay for free, and critically
            # avoids a SECOND call to detector.read_frame() here: the vision
            # thread already reads the camera at ~30Hz, and a second reader
            # racing it on the same VideoCapture is what caused the
            # "V4L2: select() timeout" warnings.
            if show:
                try:
                    frame = None
                    if status is not None and status.result is not None:
                        frame = detector.get_annotated_frame(status.result)

                    if frame is not None:
                        _draw_status_overlay(frame, status, cmd, driving if chassis is not None else None)
                        cv2.imshow("Visual Servoing + Cascade  (m=drive, q=quit)", frame)  # type: ignore

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("m") and chassis is not None:
                        driving = not driving
                        controller.set_driving(driving)  # actually gate the wheels
                        print(f"\n[mode] drive toggled: {'ON' if driving else 'OFF'}\n")

                except cv2.error:
                    print("[!] no display available — continuing text-only")
                    show = False
                except Exception as e:
                    print(f"[display] error: {e}")

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
    speed = 1.0
    if "--speed" in sys.argv:
        speed = float(sys.argv[sys.argv.index("--speed") + 1])
    use_ncnn = "--pt" not in sys.argv
    main(drive=drive, speed=speed, use_ncnn=use_ncnn)
