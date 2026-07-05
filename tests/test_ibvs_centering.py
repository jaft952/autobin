"""
Real-time IBVS centering test with live camera + YOLO11n-seg detector.

Usage:
    python test_ibvs_centering.py           -> live centering (default)
    python test_ibvs_centering.py --drive   -> live + chassis motion ON
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visual_servoing.ibvs_centering import IBVSCentering


def live(drive=False):
    """Real-time IBVS centering with live camera + YOLO11n-seg detector."""
    import cv2
    from src.perception.detector import AluminiumCanDetector

    IMGSZ = 640
    INFER_EVERY = 1
    MIN_ERROR_TO_MOVE = 0.01

    detector = AluminiumCanDetector(device="cpu", imgsz=IMGSZ, model_path="src/models/yolov11n-seg.pt") # type: ignore
    detector.start()
    centering = IBVSCentering()
    chassis = _make_chassis()
    driving = drive and chassis is not None

    print(f"\nLive IBVS centering. drive={'ON' if driving else 'OFF'}.")
    print("Keys:  m = toggle drive,  q = quit.")
    print()

    show = True
    result = None
    status = None
    i = 0

    try:
        while True:
            frame = detector.read_frame()
            if frame is None:
                continue

            if i % INFER_EVERY == 0:
                result = detector.infer(frame)
                status = centering.update(result)

                best = result.best
                if best is not None:
                    print(
                        f"center=({best.center_x:4d}, {best.center_y:4d})  "
                        f"base_center=({best.base_center[0]:4d}, {best.base_center[1]:4d})  "
                        f"size={best.width}×{best.height}  "
                        f"mask_area={best.mask_area if best.mask_area else 'None':>7}  "
                        f"err=({status.error_x:+.2f},{status.error_y:+.2f})  "
                        f"move={status.move.value:<10} aligned={status.aligned} stable={status.stable}  "
                        f"| {status.message}"
                    )

                    if driving:
                        error_mag = max(abs(status.error_x), abs(status.error_y))
                        if error_mag > MIN_ERROR_TO_MOVE:
                            chassis.drive_toward_target( # type: ignore
                                status.move, status.error_x, status.error_y
                            )
                        else:
                            chassis.stop() # type: ignore
                else:
                    print("[no detection] SEARCHING...")
                    if driving and chassis is not None:
                        chassis.stop()

            i += 1

            if not show:
                continue

            annotated = detector.get_annotated_frame(result) if result is not None else frame
            if status is not None:
                _draw_status_overlay(annotated, status, driving if chassis is not None else None)

            try:
                cv2.imshow("IBVS centering  (m=drive  q=quit)", annotated) # type: ignore
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print("[!] no display available — continuing text-only")
                show = False
                continue

            if key == ord("q"):
                break
            if key == ord("m") and chassis is not None:
                driving = not driving
                if not driving and chassis is not None:
                    chassis.stop()
                print(f"[mode] drive = {'ON' if driving else 'OFF'}")

    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        try:
            if driving and chassis is not None:
                chassis.stop()
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
            cv2.destroyAllWindows()
        except Exception:
            pass


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


if __name__ == "__main__":
    drive = "--drive" in sys.argv
    live(drive=drive)