"""
Interactive arc motion calibration tool — test real motions on the robot.

Tune arc behavior by testing different angle values and watching the robot.
Test both forward/backward arcs and stationary left/right turns.

Keys:
  1-4               : select arc direction
                      1 = ARC FORWARD-LEFT
                      2 = ARC FORWARD-RIGHT
                      3 = ARC BACKWARD-LEFT
                      4 = ARC BACKWARD-RIGHT
  5/6               : stationary TURN LEFT / TURN RIGHT

  [/]               : fine tune angle (+/- 1°, arc modes only)
  up/down           : adjust angle (+/- 5°, arc modes only)
  r                 : reset to 90° (sharp turn, arc modes only)

  +/-               : speed up/down (0.05 steps)

  ENTER / SPACE     : run motion for 0.3s (tests it on the robot)
  a                 : apply motion continuously (hold until key press or 10s max)

  p                 : print detailed wheel values (console only)
  x                 : quit

The angle controls the inner wheel ratio (arc modes):
  0°   = straight (inner wheel at full speed)
  90°  = sharp turn (inner wheel at ~33%)
  180° = spin in place (inner wheel at 0)
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motion.differential_kinematics import DifferentialKinematics
from src.motion.calibration import MotionCalibration

# Cross-platform non-blocking single-key reader
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


def _format_wheel_output(name: str, cmd, speed: float) -> str:
    """Format a wheel command for display."""
    return (f"[{name:20}] L={cmd.left_speed:+7.2f} R={cmd.right_speed:+7.2f} "
            f"trim={cmd.apply_trim}/{cmd.trim_set} (speed={speed:.2f})")


def main():
    try:
        cal = MotionCalibration()
        kin = DifferentialKinematics(cal)
    except Exception as exc:
        print(f"[!] calibration not available: {exc}")
        return

    actuator = None
    try:
        from src.hardware.actuators.pwm_driver import PWMActuator
        actuator = PWMActuator(calibration=cal)
        print("[✓] motor driver ready")
    except Exception as exc:
        print(f"[!] motor driver not available ({exc}); console-only mode")

    angle = 90.0  # start at sharp turn
    speed = 0.50
    mode = 1  # 1=arc_forward_left, 2=arc_forward_right, 3=arc_backward_left, 4=arc_backward_right, 5=turn_left, 6=turn_right
    mode_names = {
        1: "ARC FWD-LEFT",
        2: "ARC FWD-RIGHT",
        3: "ARC BACK-LEFT",
        4: "ARC BACK-RIGHT",
        5: "TURN LEFT",
        6: "TURN RIGHT",
    }

    jog_s = 0.30  # burst duration for ENTER/SPACE

    print(__doc__)
    print(f"\nCalibration:")
    print(f"  forward_speed={cal.forward_speed}")
    print(f"  arc_speed={cal.arc_speed}")
    print(f"  turn_speed={cal.turn_speed}")
    print(f"  arc_inner_wheel_ratio={cal.arc_inner_wheel_ratio}")
    print(f"  motor_a_forward_trim={cal.motor_a_forward_trim}")
    print(f"  motor_b_forward_trim={cal.motor_b_forward_trim}")
    print(f"  motor_a_backward_trim={cal.motor_a_backward_trim}")
    print(f"  motor_b_backward_trim={cal.motor_b_backward_trim}")
    print()

    def get_current_command():
        """Get the current wheel command based on mode and angle.

        NOTE: mode 1/2 and 3/4 call the OPPOSITE-named kinematics function
        on purpose — on this robot's wiring, arc_forward_left() actually
        drives a rightward arc (and vice versa). differential_kinematics.py
        is shared by other modules so it's left as-is; this swap lives only
        here so the on-screen label matches what the robot actually does."""
        if mode == 1:
            return kin.arc_forward_right(angle, cal.arc_speed * speed)
        elif mode == 2:
            return kin.arc_forward_left(angle, cal.arc_speed * speed)
        elif mode == 3:
            return kin.arc_backward_right(angle, cal.arc_speed * speed)
        elif mode == 4:
            return kin.arc_backward_left(angle, cal.arc_speed * speed)
        elif mode == 5:
            return kin.turn_left(cal.turn_speed * speed)
        else:  # mode == 6
            return kin.turn_right(cal.turn_speed * speed)

    def print_current():
        """Print current wheel command."""
        cmd = get_current_command()
        if mode in (1, 2, 3, 4):
            print(f"Angle: {angle:6.1f}° | {_format_wheel_output(mode_names[mode], cmd, speed)}")
        else:  # stationary turns
            print(f"          | {_format_wheel_output(mode_names[mode], cmd, speed)}")

    def run_motion(duration: float):
        """Apply the current motion to the wheels for the given duration."""
        if actuator is None:
            print("[!] no motor driver — cannot run motion")
            return
        cmd = get_current_command()
        print(f"[RUN {duration:.2f}s] {mode_names[mode]}" + (f" @ {angle:.1f}°" if mode in (1, 2, 3, 4) else ""))
        actuator.apply(cmd)
        time.sleep(duration)
        actuator.stop()
        print("[STOP]")

    print_current()

    try:
        while True:
            key = get_key(timeout=0.1)
            if key is None:
                continue

            key = key.lower()

            # Exit
            if key == "x":
                break

            # Mode selection (1-6)
            elif key in ("1", "2", "3", "4", "5", "6"):
                mode = int(key)
                print(f"\n[MODE] switched to {mode_names[mode]}")
                if mode not in (5, 6):
                    print(f"       (angle controls the arc curvature: 0°=straight, 90°=sharp, 180°=spin)")
                print()
                print_current()
                continue

            # Angle adjustment (only for arc modes)
            elif key in ("up", "down", "[", "]") and mode in (1, 2, 3, 4):
                if key == "up":
                    angle = min(180.0, angle + 5.0)
                elif key == "down":
                    angle = max(0.0, angle - 5.0)
                elif key == "[":
                    angle = max(0.0, angle - 1.0)
                elif key == "]":
                    angle = min(180.0, angle + 1.0)
                print_current()
                continue

            # Arrow keys for UP/DOWN (need special handling on Windows)
            elif key == "\x00" or key == "\xe0":  # escape sequence start (Windows)
                # Try to read the next key
                key2 = get_key(timeout=0.05)
                if key2 == "H" and mode in (1, 2, 3, 4):  # UP arrow
                    angle = min(180.0, angle + 5.0)
                    print_current()
                elif key2 == "P" and mode in (1, 2, 3, 4):  # DOWN arrow
                    angle = max(0.0, angle - 5.0)
                    print_current()
                continue

            # Speed adjustment
            elif key in ("+", "="):
                speed = min(1.0, speed + 0.05)
                print(f"[SPEED] {speed:.2f}")
                print_current()
                continue
            elif key in ("-", "_"):
                speed = max(0.05, speed - 0.05)
                print(f"[SPEED] {speed:.2f}")
                print_current()
                continue

            # Run motion (jog-style: short burst)
            elif key in (" ", "\r", "\n"):  # SPACE or ENTER
                run_motion(jog_s)
                print_current()
                continue

            # Run motion (continuous: hold for up to 10 seconds)
            elif key == "a":
                if actuator is None:
                    print("[!] no motor driver — cannot run motion")
                    continue
                cmd = get_current_command()
                print(f"[RUN CONTINUOUS] {mode_names[mode]}" + (f" @ {angle:.1f}°" if mode in (1, 2, 3, 4) else ""))
                print("  (press any key to stop, or auto-stops after 10s)")
                actuator.apply(cmd)
                t0 = time.time()
                while time.time() - t0 < 10.0:
                    if get_key(timeout=0.1) is not None:
                        break
                actuator.stop()
                print("[STOP]")
                print_current()
                continue

            # Print detailed info
            elif key == "p":
                cmd = get_current_command()
                if mode in (1, 2, 3, 4):
                    print(f"\nDetailed {mode_names[mode]} @ {angle:.1f}°:")
                else:
                    print(f"\nDetailed {mode_names[mode]}:")
                print(f"  Speed multiplier: {speed:.2f}")
                print(f"  Left speed:  {cmd.left_speed:+.3f}")
                print(f"  Right speed: {cmd.right_speed:+.3f}")
                print(f"  Apply trim:  {cmd.apply_trim}")
                print(f"  Trim set:    {cmd.trim_set}")
                if mode in (1, 2, 3, 4):
                    from src.motion.differential_kinematics import _ratio_from_angle
                    ratio = _ratio_from_angle(angle)
                    print(f"  Angle:       {angle:.1f}°")
                    print(f"  Inner ratio: {ratio:.3f}")
                print()
                print_current()
                continue

            # Reset angle
            elif key == "r" and mode in (1, 2, 3, 4):
                angle = 90.0
                print(f"[ANGLE] reset to 90° (sharp turn)")
                print_current()
                continue

    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if actuator is not None:
            try:
                actuator.stop()
                actuator.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
