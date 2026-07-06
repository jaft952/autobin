"""
Real-time Visual Servoing with Cascade Control (multi-rate, smooth motion).

Architecture:
  - Vision thread: ~30 Hz (YOLO11n-seg detection + IBVS error computation)
  - Motor thread: ~1000 Hz (smooth interpolation + velocity limiting)
  - Non-blocking synchronization via TrackingBuffer
  - Dynamic filter tuning based on mask_area confidence

Usage:
    python test_ibvs_centering.py           -> cascade control (default)
    python test_ibvs_centering.py --drive   -> cascade control + chassis motion ON
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visual_servoing.ibvs_centering import IBVSCentering
from src.visual_servoing.cascade_controller import CascadeController




def _draw_status_overlay(frame, status, driving=None):
    """Draw IBVS status on frame."""
    import cv2
    if frame is None:
        return
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    action = status.move.value.upper()
    color = (0, 255, 0) if status.stable else (0, 165, 255)
    cv2.putText(frame, f"{action}  aligned={status.aligned}  stable={status.stable}",
                (20, fh - 25), font, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, f"{action}  aligned={status.aligned}  stable={status.stable}",
                (20, fh - 25), font, 0.8, color, 2)

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


def main(drive=False):
    """Real-time visual servoing with cascade control (multi-rate, smooth motion)."""
    import cv2
    from src.perception.detector import AluminiumCanDetector

    IMGSZ = 640

    detector = AluminiumCanDetector(device="cpu", imgsz=IMGSZ, model_path="src/models/yolov11n-seg.pt")  # type: ignore
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
    print(f"Keys: m = toggle drive, q = quit")
    print(f"{'='*70}\n")

    # Create cascade controller (non-blocking threads)
    controller = CascadeController(detector, centering, chassis)
    controller.start()

    show = True
    result = None
    status = None

    try:
        while True:
            # Get latest vision status (non-blocking)
            status = controller.get_status()

            if status:
                # Attempt to get detection for visualization
                try:
                    frame = detector.read_frame()
                    if frame is not None:
                        result = detector.infer(frame)
                except Exception:
                    pass

                # Display status
                print(
                    f"err=({status.error_x:+.3f},{status.error_y:+.3f})  "
                    f"mask_area={status.mask_area if status.mask_area else 'None':>7}  "
                    f"quality={status.quality():.2f}  "
                    f"aligned={status.aligned}  stable={status.stable}  "
                    f"drive={'ON' if driving else 'OFF'}"
                )

                # Visualization
                if show and result and result.best:
                    annotated = detector.get_annotated_frame(result)
                    _draw_status_overlay(annotated, status, driving if chassis is not None else None)

                    try:
                        cv2.imshow("Visual Servoing + Cascade  (m=drive, q=quit)", annotated)  # type: ignore
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord("q"):
                            break
                        if key == ord("m") and chassis is not None:
                            driving = not driving
                            print(f"\n[mode] drive toggled: {'ON' if driving else 'OFF'}\n")

                    except cv2.error:
                        print("[!] no display available — continuing text-only")
                        show = False
                        continue
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
    main(drive=drive)
