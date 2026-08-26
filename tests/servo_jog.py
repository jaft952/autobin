"""
Servo jog & pose capture — line-based, works on the Pi (and Windows sim).

Hand-jog the arm to a fixed pose (e.g. the bin drop position), then PRINT/SAVE
the exact servo angles. Uses ArmActuator and the shared arm_settings profile,
so pulse calibration, angle clamps and pacing match the real system — captured
angles drop straight into arm_settings.

Commands (type, then Enter):
  2 160        set CH2 to 160 deg  (channel 1-16, angle 0-180)
  +2 / -2      nudge the LAST-touched channel by +/- step (default 5 deg)
  step 2       change the nudge step to 2 deg
  speed 2 0.5  move pacing: 2 deg per 0.5 s
  p            PRINT all 6 current angles (and where the model thinks the tip is)
  s name       SAVE current angles to tests/captured_poses.txt under 'name'
  r            RELEASE all servos — cut PWM so the arm goes LIMP
  h            home CH1-6 to neutral
  q            quit (servos left where they are)

Run:  python tests/servo_jog.py

NOTHING moves on startup — press 'h' to home. These servos have no position
feedback, so an un-jogged channel ramps from the ASSUMED neutral (home first
if unsure).
"""
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.arm_settings import GRIPPER_OPEN, SERVO_NEUTRAL_CMD, jog_profile
from src.hardware.actuators.pca9685_driver import (
    ArmActuator, stepped_move, SERVO_RANGE_DEG, clamp_channel_angle,
)

# CH1-5 from the calibrated model-zero; CH6 homes to the system's open angle
# (the model's own gripper neutral predates the MG996R and sits past its stop).
NEUTRAL = [SERVO_NEUTRAL_CMD[i] for i in range(1, 7)]
NEUTRAL[5] = GRIPPER_OPEN
NUM_CH = 16   # jog any PCA9685 channel (hardware testing)
POSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_poses.txt")


def main():
    act = ArmActuator()
    profile = jog_profile()
    angles = list(NEUTRAL) + [90.0] * (NUM_CH - len(NEUTRAL))  # CH7-16 default to mid
    last_ch = 0
    step = 5.0

    def cur(ch):
        """Last-commanded angle (hardware truth), falling back to the tracked
        value when the channel was never set / was released."""
        a = act.kit.servo[ch].angle
        return float(a) if a is not None else float(angles[ch])

    def apply(i, val):
        """Move CH(i+1) to val, ramping from the servo's ACTUAL current angle.
        ALL channels step gently — an instant gripper move spikes current and
        can brown out / drop the arm."""
        val = clamp_channel_angle(i, float(val))   # per-channel safe window
        start = [cur(c) for c in range(NUM_CH)]
        target = list(start)
        target[i] = val
        stepped_move(act, start, target, profile.step_deg, profile.step_delay)
        angles[:] = start
        angles[i] = val
        print(f"  CH{i + 1} = {val:.1f} deg")

    def print_pose():
        print("\n  current servo angles (CH1-7):")
        print("   ", [round(a, 1) for a in angles[:7]])
        print("    arm only (CH1-5):", [round(a, 1) for a in angles[:5]])
        print()

    print("=" * 60)
    print(" SERVO JOG & POSE CAPTURE")
    print(" type 'h' to home, '2 160' to set a servo, 'p' to print, 'q' to quit")
    print(f" pacing {profile.step_deg} deg / {profile.step_delay}s "
          f"= {profile.speed_dps:.1f} deg/s ('speed D S' to change)")
    print("=" * 60)
    print_pose()

    while True:
        try:
            raw = input("jog> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "q":
            break
        elif cmd == "r":
            act.release()
            print("  RELEASED all servos (no signal — arm is limp). "
                  "Set any channel to re-engage.")
        elif cmd == "h":
            # One channel at a time, CH1 -> CH6, then force-write neutral so a
            # channel goes home even if the tracked pose already matched it.
            print("  homing CH1 -> CH6 in sequence...")
            for i, a in enumerate(NEUTRAL):
                apply(i, a)
                a = clamp_channel_angle(i, a)      # never bypass the window
                act.kit.servo[i].angle = a # type: ignore
                angles[i] = a
            print(f"  homed CH1-6 -> {[round(a, 1) for a in NEUTRAL]}")
            last_ch = 0
        elif cmd == "p":
            print_pose()
        elif cmd == "step" and len(parts) == 2:
            try:
                step = float(parts[1])
                print(f"  nudge step = {step} deg")
            except ValueError:
                print("  usage: step <degrees>")
        elif cmd == "speed":
            try:
                profile.set_speed(float(parts[1]), float(parts[2]))
                print(f"  pacing {profile.step_deg} deg / {profile.step_delay}s "
                      f"= {profile.speed_dps:.1f} deg/s")
            except (IndexError, ValueError):
                print("  usage: speed <degrees> <seconds>")
        elif cmd == "s":
            name = parts[1] if len(parts) > 1 else "unnamed"
            with open(POSE_FILE, "a", encoding="utf-8") as f:
                f.write(f"{name}\t{datetime.now():%Y-%m-%d %H:%M}\t"
                        f"{[round(a, 1) for a in angles]}\n")
            print(f"  saved '{name}' -> {POSE_FILE}")
        elif cmd.startswith(("+", "-")) and len(cmd) > 1 and cmd[1:].isdigit():
            ch = int(cmd[1:])
            if 1 <= ch <= NUM_CH:
                last_ch = ch - 1
                delta = step if cmd[0] == "+" else -step
                apply(last_ch, angles[last_ch] + delta)
            else:
                print(f"  channel must be 1-{NUM_CH}")
        elif len(parts) == 2 and parts[0].isdigit():
            ch = int(parts[0])
            if 1 <= ch <= NUM_CH:
                try:
                    last_ch = ch - 1
                    apply(last_ch, float(parts[1]))
                except ValueError:
                    print(f"  angle must be a number 0-{SERVO_RANGE_DEG}")
            else:
                print(f"  channel must be 1-{NUM_CH}")
        else:
            print("  commands: 'CH angle' | '+CH'/'-CH' | 'step N' | 'speed D S' "
                  "| p | s name | h | r | q")

    print("done (servos still holding — press 'r' before 'q' to leave the arm limp).")


if __name__ == "__main__":
    main()
