"""
tests/capture_training_images.py

Capture training photos for the YOLO tin detector using the webcam — no YOLO.

    python tests/capture_training_images.py
    python tests/capture_training_images.py /media/homa/HOMA/tintrain   # custom dir

Opens the camera and shows a live preview. Keys:
    s  -> save the current frame as a JPG into the output folder
    q  -> quit

Default output folder is the pendrive at /media/homa/HOMA/tintrain. If that
pendrive isn't mounted the tool refuses to run (so it never silently fills the
SD card) — plug it in, or pass your own folder as an argument.

Files are named tin_0001.jpg, tin_0002.jpg, ... continuing past whatever is
already in the folder, so you can collect across many sessions without
overwriting. Saved images are the CLEAN camera frame (the on-screen text is only
drawn on the preview, never on the file). Capture is at 1280x720 to match the
detector's runtime resolution. Run it on the Pi desktop (needs a display).
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.perception.detector import open_camera_capture

DEFAULT_OUT = "/media/homa/HOMA/tintrain"
W, H = 1280, 720


def _next_index(out_dir: Path) -> int:
    """Next tin_NNNN index, continuing past any existing tin_*.jpg in out_dir."""
    existing = []
    for p in out_dir.glob("tin_*.jpg"):
        stem = p.stem.split("_")[-1]
        if stem.isdigit():
            existing.append(int(stem))
    return (max(existing) + 1) if existing else 1


def main():
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    else:
        out_dir = Path(DEFAULT_OUT)
        # Don't silently write to the SD card if the pendrive isn't mounted.
        if not out_dir.parent.exists():
            print(f"✗ Pendrive not found: {out_dir.parent}")
            print("  Plug in / mount the pendrive, or pass a folder, e.g.:")
            print("    python tests/capture_training_images.py ~/tintrain")
            return
    out_dir.mkdir(parents=True, exist_ok=True)

    cap, fw, fh, fps = open_camera_capture(0, W, H)
    print(f"✓ Camera opened: {fw}x{fh} @ {fps:.0f}FPS")
    print(f"✓ Saving to: {out_dir}")
    print("\n  s = capture,  q = quit\n")

    idx = _next_index(out_dir)
    saved = 0
    flash = 0  # frames remaining to show the "SAVED" banner
    win = "capture training images  (s=save  q=quit)"
    cv2.namedWindow(win)
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠ failed to read frame from camera")
                break

            display = frame.copy()  # draw HUD on the copy; save the clean frame
            cv2.putText(display, f"saved this session: {saved}", (15, 35),
                        font, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"next: tin_{idx:04d}.jpg", (15, 70),
                        font, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "s = capture    q = quit", (15, fh - 20),
                        font, 0.7, (255, 255, 0), 2)
            if flash > 0:
                cv2.putText(display, "SAVED", (fw // 2 - 95, fh // 2),
                            font, 2.0, (0, 255, 0), 4)
                flash -= 1

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = out_dir / f"tin_{idx:04d}.jpg"
                if cv2.imwrite(str(path), frame):
                    print(f"✓ saved {path}")
                    idx += 1
                    saved += 1
                    flash = 12
                else:
                    print(f"✗ failed to write {path} (disk full / not writable?)")
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {saved} image(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
