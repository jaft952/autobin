"""
tests/capture_training_images.py

Capture training photos for the YOLO tin detector using the webcam — no YOLO.

    python tests/capture_training_images.py
    python tests/capture_training_images.py /media/homa/HOMA/tintrain   # custom dir

Opens the camera and shows a live preview. Keys:
    s  -> save the current frame as a JPG into the output folder
    r  -> start/stop recording a video clip (press once to start, again to stop)
    q  -> quit

Default output folder is the pendrive at /media/homa/HOMA/tintrain. If that
pendrive isn't mounted the tool refuses to run (so it never silently fills the
SD card) — plug it in, or pass your own folder as an argument.

Photos are named tin_0001.jpg, tin_0002.jpg, ...; videos vid_0001.mp4 (or .avi if
the mp4 codec isn't available), both continuing past whatever is already in the
folder so you can collect across many sessions without overwriting. Saved images
and recorded video are the CLEAN camera frame (the on-screen text is only drawn on
the preview, never on the file). Capture is at 1280x720 to match the detector's
runtime resolution. Run it on the Pi desktop (needs a display).
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


def _next_video_index(out_dir: Path) -> int:
    """Next vid_NNNN index across .mp4 and .avi clips in out_dir."""
    nums = []
    for p in list(out_dir.glob("vid_*.mp4")) + list(out_dir.glob("vid_*.avi")):
        s = p.stem.split("_")[-1]
        if s.isdigit():
            nums.append(int(s))
    return (max(nums) + 1) if nums else 1


def _open_writer(out_dir: Path, fw: int, fh: int, fps: float):
    """Open a VideoWriter for the next vid_NNNN clip. Try mp4 (mp4v) first, fall
    back to avi (MJPG) if this OpenCV build can't encode mp4. Returns (writer,
    path) or (None, None) if no codec works."""
    idx = _next_video_index(out_dir)
    rate = fps if 1.0 <= fps <= 120.0 else 30.0   # guard against a bogus reported fps
    for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        path = out_dir / f"vid_{idx:04d}{ext}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                                 rate, (fw, fh))
        if writer.isOpened():
            return writer, path
        writer.release()
    return None, None


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
    print("\n  s = capture,  r = record start/stop,  q = quit\n")

    idx = _next_index(out_dir)
    saved = 0
    flash = 0  # frames remaining to show the "SAVED" banner
    recording = False      # toggled by 'r'
    writer = None          # cv2.VideoWriter while recording
    video_path = None
    vid_frames = 0
    win = "capture training images  (s=save  r=rec  q=quit)"
    cv2.namedWindow(win)
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠ failed to read frame from camera")
                break

            if recording and writer is not None:   # record the CLEAN frame
                writer.write(frame)
                vid_frames += 1

            display = frame.copy()  # draw HUD on the copy; save the clean frame
            cv2.putText(display, f"saved this session: {saved}", (15, 35),
                        font, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"next: tin_{idx:04d}.jpg", (15, 70),
                        font, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "s = capture   r = record   q = quit", (15, fh - 20),
                        font, 0.7, (255, 255, 0), 2)
            if recording:   # red dot + frame count, top-right
                cv2.circle(display, (fw - 245, 30), 10, (0, 0, 255), -1)
                cv2.putText(display, f"REC  {vid_frames}f", (fw - 225, 38),
                            font, 0.8, (0, 0, 255), 2)
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
            if key == ord("r"):
                if not recording:
                    writer, video_path = _open_writer(out_dir, fw, fh, fps)
                    if writer is None:
                        print("✗ could not open a video writer (no codec available "
                              "for mp4 or avi).")
                    else:
                        recording, vid_frames = True, 0
                        print(f"● recording -> {video_path}")
                else:
                    recording = False
                    writer.release()
                    print(f"✓ saved video {video_path}  ({vid_frames} frames)")
                    writer, video_path = None, None
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if writer is not None:   # quit while still recording -> finalize the clip
            writer.release()
            print(f"✓ saved video {video_path}  ({vid_frames} frames)")
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {saved} image(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
