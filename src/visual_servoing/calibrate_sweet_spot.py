"""
src/visual_servoing/calibrate_sweet_spot.py

Sweet-spot (grasp center) calibrator — camera only, NO YOLO.

The camera is fixed to the arm base, so a tin placed at the arm's comfortable
grasp position always lands at the same pixel. This tool finds that pixel:

    python src/visual_servoing/calibrate_sweet_spot.py                  # UPRIGHT (default)
    python src/visual_servoing/calibrate_sweet_spot.py --lying --point  # LYING, point mode
    python src/visual_servoing/calibrate_sweet_spot.py --lying --line   # LYING, line mode

Run it ON THE PI DESKTOP (it needs a display). Put a tin where the arm grasps it
best, LEFT-CLICK the point the centering should aim at, then press 's' to save:
  - default (upright): click the tin's BASE   -> saves target_x / target_y
  - --lying:           click the tin's CENTER -> saves lying_target_x / lying_target_y
  - --lying --point / --line also saves lying_mode (how a lying tin counts as
    aligned: "point" = center reaches (x,y); "line" = only the Y/distance matters,
    left/right ignored). Omit both to leave the saved lying_mode unchanged.
Everything lives in config/centering_config.yaml (other keys preserved), so the
upright and lying targets coexist. Press 'q' to quit.

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
from src.visual_servoing.ibvs_centering import CENTERING_CONFIG_PATH

W, H = 1280, 720  # must match the detector's capture resolution (see module docstring)


def _read_yaml() -> dict:
    import yaml
    if CENTERING_CONFIG_PATH.exists():
        return yaml.safe_load(CENTERING_CONFIG_PATH.read_text()) or {}
    return {}


def _save(kx: str, ky: str, x: float, y: float, extra: dict = None):
    """Write the two sweet-spot keys (plus any `extra` keys) into
    centering_config.yaml, preserving everything else already there (so upright and
    lying targets coexist)."""
    import yaml
    CENTERING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _read_yaml()
    data[kx] = round(float(x), 4)
    data[ky] = round(float(y), 4)
    if extra:
        data.update(extra)
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

    # Mode: default = UPRIGHT tin base; --lying = LYING tin center. Each saves its
    # own keys so both targets coexist in the YAML.
    lying = "--lying" in sys.argv
    # Lying alignment mode to SAVE: --line or --point (only meaningful with --lying).
    # Omit both -> leave whatever lying_mode is already in the yaml.
    mode = "line" if "--line" in sys.argv else "point" if "--point" in sys.argv else None
    kx, ky = ("lying_target_x", "lying_target_y") if lying else ("target_x", "target_y")
    point_name = "CENTER" if lying else "BASE"
    data = _read_yaml()
    sx = float(data.get(kx, 0.5))
    sy = float(data.get(ky, 0.85 if lying else 0.5))
    saved_px = (int(sx * fw), int(sy * fh))

    win = (f"sweet-spot calibrator [{'LYING' if lying else 'UPRIGHT'}]"
           "  (click, s=save, q=quit)")
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    modestr = f" [{mode} mode]" if (lying and mode) else ""
    print(f"\nMode: {'LYING' if lying else 'UPRIGHT'} tin{modestr}. Put a "
          f"{'lying' if lying else 'upright'} tin at the grasp position, LEFT-CLICK "
          f"its {point_name}, then 's' to save (-> {kx}/{ky}). 'q' to quit.\n")
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

            # Currently saved target (yellow). In lying mode also draw the saved Y
            # as a horizontal "line target" (line mode uses only the Y).
            if lying:
                cv2.line(frame, (0, saved_px[1]), (fw, saved_px[1]), (0, 200, 200), 1)
            cv2.drawMarker(frame, saved_px, (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(frame, f"saved {'lying' if lying else 'upright'} target",
                        (saved_px[0] + 12, saved_px[1] - 12),
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
                extra = {"lying_mode": mode} if (lying and mode) else None
                _save(kx, ky, nx, ny, extra)
                saved_px = (click[0], click[1])
                modesave = f"  lying_mode={mode}" if (lying and mode) else ""
                print(f"✓ saved {'LYING' if lying else 'UPRIGHT'} sweet spot  "
                      f"{kx}={nx:.4f} {ky}={ny:.4f}{modesave} -> {CENTERING_CONFIG_PATH}")
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
