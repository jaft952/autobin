"""
Manual differential-drive test with keyboard control.

Drives the chassis directly through ChassisController.set_motor_pwm() —
the exact same call the cascade controller uses — so you can verify:
  - forward/backward polarity
  - steering (left/right) direction
  - minimum speed needed to actually move on the floor

Usage:
    python tests/test_differential_drive.py

Keys:
  w : forward              s : backward
  a : turn left            d : turn right
  q : arc forward-left     e : arc forward-right
  z : arc backward-left    c : arc backward-right
  space : stop
  + / - : speed up / down (step 0.05)
  i : invert forward polarity (if w drives backward)
  k : invert steer polarity  (if a turns right)
  x : quit
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------------
# Cross-platform non-blocking single-key reader
# ----------------------------------------------------------------------
if os.name == "nt":
    import msvcrt

    def get_key(timeout: float = 0.1):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    return ch.decode()
                except UnicodeDecodeError:
                    return None
            time.sleep(0.01)
        return None
else:
    import termios
    import tty
    import select

    def get_key(timeout: float = 0.1):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            if r:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ----------------------------------------------------------------------
# Key -> (forward, steer) in "intent" units, before polarity/speed scaling.
# forward: +1 = drive toward the tin ("forward"), -1 = reverse
# steer:   -1 = turn left, +1 = turn right
# ----------------------------------------------------------------------
KEY_MAP = {
    "w": ("FORWARD", 1.0, 0.0),
    "s": ("BACKWARD", -1.0, 0.0),
    "a": ("TURN LEFT", 0.0, -1.0),
    "d": ("TURN RIGHT", 0.0, 1.0),
    "q": ("ARC FWD-LEFT", 1.0, -0.5),
    "e": ("ARC FWD-RIGHT", 1.0, 0.5),
    "z": ("ARC BACK-LEFT", -1.0, -0.5),
    "c": ("ARC BACK-RIGHT", -1.0, 0.5),
    " ": ("STOP", 0.0, 0.0),
}


def main():
    from src.visual_servoing.chassis_controller import ChassisController

    try:
        chassis = ChassisController()
    except Exception as exc:
        print(f"[!] chassis not available: {exc}")
        return

    speed = 0.50        # command magnitude (0..1)
    fwd_sign = -1.0     # set_motor_pwm convention: negative = forward
    steer_sign = 1.0

    print(__doc__)
    print(f"speed={speed:.2f}  fwd_sign={fwd_sign:+.0f}  steer_sign={steer_sign:+.0f}")
    print("ready — press keys to drive.\n")

    current = ("STOP", 0.0, 0.0)

    try:
        while True:
            key = get_key(timeout=0.1)
            if key is None:
                continue

            key = key.lower()

            if key == "x":
                break

            if key in ("+", "="):
                speed = min(1.0, speed + 0.05)
                print(f"[speed] {speed:.2f}")
                key = None
            elif key in ("-", "_"):
                speed = max(0.05, speed - 0.05)
                print(f"[speed] {speed:.2f}")
                key = None
            elif key == "i":
                fwd_sign = -fwd_sign
                print(f"[polarity] forward sign = {fwd_sign:+.0f}")
                key = None
            elif key == "k":
                steer_sign = -steer_sign
                print(f"[polarity] steer sign = {steer_sign:+.0f}")
                key = None

            if key is not None and key in KEY_MAP:
                current = KEY_MAP[key]

            # (Re)apply current motion with latest speed/polarity so that
            # speed and polarity changes take effect immediately.
            name, f_intent, s_intent = current
            forward = fwd_sign * f_intent * speed
            steer = steer_sign * s_intent * speed

            if name == "STOP":
                chassis.stop()
                print("STOP")
            else:
                chassis.set_motor_pwm(forward, steer)
                # Mirror the wheel math in set_motor_pwm (camera-backward config)
                left = forward + steer
                right = forward - steer
                print(
                    f"{name:15}  forward={forward:+.2f}  steer={steer:+.2f}  "
                    f"->  left={left:+.2f}  right={right:+.2f}  (speed={speed:.2f})"
                )

    except KeyboardInterrupt:
        pass
    finally:
        try:
            chassis.stop()
        except Exception:
            pass
        try:
            chassis.close()
        except Exception:
            pass
        print("\n[stopped, GPIO released]")


if __name__ == "__main__":
    main()
