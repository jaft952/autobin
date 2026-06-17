"""
src/visual_servoing/calibrate_sweet_spot.py

Sweet-spot (grasp center) calibrator — camera only, NO YOLO.

The camera is fixed to the arm base, so a tin placed at the arm's comfortable
grasp position always lands at the same pixel. This tool finds that pixel:

    python src/visual_servoing/calibrate_sweet_spot.py

Run it ON THE PI DESKTOP (it needs a display). Put a tin where the arm grasps it
best, LEFT-CLICK the tin's center in the window, then press 's' to save. The
normalized coords are written to config/centering_config.yaml, which
IBVSCentering loads automatically. Press 'q' to quit.

No model is loaded and no inference runs, so it stays smooth even on a Pi 4.
The capture resolution matches the detector's (1280x720) on purpose: the C270
crops differently per resolution, so a sweet spot calibrated here only transfers
to live detection when both use the same resolution.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2

from src.perception.detector import open_camera_capture
from src.visual_servoing.ibvs_centering import CenteringConfig, CENTERING_CONFIG_PATH

W, H = 1280, 720  # must match the detector's capture resolution (see module docstring)


def _save(target_x: float, target_y: float):
    """Write target_x / target_y into centering_config.yaml, preserving any
    other keys already in the file."""
    import yaml
    CENTERING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if CENTERING_CONFIG_PATH.exists():
        data = yaml.safe_load(CENTERING_CONFIG_PATH.read_text()) or {}
    data["target_x"] = round(float(target_x), 4)
    data["target_y"] = round(float(target_y), 4)
    CENTERING_CONFIG_PATH.write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    )


def main():
    state = {"click": None, "mouse": (0, 0)}  # last left-click pixel; current mouse pixel

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            state["mouse"] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
            print(f"clicked  center=({x},{y})  norm=({x / fw:.4f},{y / fh:.4f})")

    cap, fw, fh, fps = open_camera_capture(0, W, H)
    print(f"✓ Camera opened: {fw}x{fh} @ {fps:.0f}FPS")
    if (fw, fh) != (W, H):
        print(f"⚠ camera resolution {fw}x{fh} != expected {W}x{H}; the saved "
              f"sweet spot is normalized to {fw}x{fh}. Make the detector use the "
              f"same resolution.")

    # Current saved target (from YAML if it exists), shown for reference.
    cfg = CenteringConfig.load()
    saved_px = (int(cfg.target_x * fw), int(cfg.target_y * fh))

    win = "sweet-spot calibrator  (click tin, 's' save, 'q' quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    print("\nPut a tin at the arm's grasp position, LEFT-CLICK its center, "
          "then press 's' to save. 'q' to quit.\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠ failed to read frame from camera")
                break

            # Faint image-center reference cross.
            ccx, ccy = fw // 2, fh // 2
            cv2.line(frame, (ccx, 0), (ccx, fh), (90, 90, 90), 1)
            cv2.line(frame, (0, ccy), (fw, ccy), (90, 90, 90), 1)

            # Currently saved target (yellow).
            cv2.drawMarker(frame, saved_px, (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(frame, "saved target", (saved_px[0] + 12, saved_px[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Clicked candidate (red).
            click = state["click"]
            if click is not None:
                cv2.drawMarker(frame, click, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 2)
                cv2.circle(frame, click, 5, (0, 0, 255), -1)
                cv2.putText(frame,
                            f"click=({click[0]},{click[1]})  "
                            f"norm=({click[0] / fw:.3f},{click[1] / fh:.3f})  "
                            f"[press s to save]",
                            (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

            # Live mouse readout (top-left).
            mx, my = state["mouse"]
            cv2.putText(frame, f"mouse=({mx},{my})  norm=({mx / fw:.3f},{my / fh:.3f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if click is None:
                    print("nothing to save yet — click the tin first.")
                    continue
                nx, ny = click[0] / fw, click[1] / fh
                _save(nx, ny)
                saved_px = (click[0], click[1])
                print(f"✓ saved sweet spot  norm=({nx:.4f},{ny:.4f}) "
                      f"-> {CENTERING_CONFIG_PATH}")
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
