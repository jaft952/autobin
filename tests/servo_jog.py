"""
Servo jog & pose capture — line-based, works on the Pi (and Windows sim).

Purpose: hand-jog the arm to a fixed pose (e.g. the trash-bin drop position that
sits on the chassis), then PRINT/SAVE the exact servo angles so you can hardcode
them. Uses ArmActuator, so the 270° actuation_range + pulse-width calibration are
the SAME as the real system — the angles you read here are directly usable in
GraspPlanner.set_arm_angles().

Commands (type, then Enter):
  2 160        set CH2 to 160°   (channel 1-16, angle 0-270) — e.g. move the
               gripper plug to a spare channel to test if a fault is the servo
               or the PCA9685 channel
  +2 / -2      nudge the LAST-touched channel by +/- step (default 5°)
  step 2       change the nudge step to 2°
  p            PRINT all 6 current angles (and where the model thinks the tip is)
  s name       SAVE current angles to tests/captured_poses.txt under 'name'
  r            RELEASE all servos — cut PWM so the arm goes LIMP (stops twitching,
               and won't snap back when servo power returns). Set a channel to re-engage.
  h            home all to neutral (135, gripper 120)
  q            quit (servos left where they are)

Run:  python tests/servo_jog.py

Note: each move is SLOW/stepped (same gentle speed as the grasp in
test_ibvs_centering.py), not an instant jump — tune STEP_DEG / STEP_DELAY below.
"""
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.actuators.pca9685_driver import ArmActuator, stepped_move
from src.arm.kinematics import SERVO_NEUTRAL_CMD

# Forward kinematics is optional (needs ikpy). If unavailable we still jog/print.
try:
    from src.arm.kinematics import ArmKinematics
    _kin = ArmKinematics()
except Exception as e:  # pragma: no cover
    _kin = None
    print(f"[note] forward-kinematics preview disabled ({e})")

# Neutral commands (actuation_range=180): CH1-5 from the calibrated model-zero, CH6 gripper.
NEUTRAL = [SERVO_NEUTRAL_CMD[i] for i in range(1, 7)]  # CH1-5 arm, CH6 gripper
NUM_CH = 16   # PCA9685 has 16 channels — allow jogging any of them (hardware testing)
POSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_poses.txt")

# Slow, stepped servo motion — same speed as the grasp in tests/test_ibvs_centering.py
# (GRASP_STEP_DEG / GRASP_STEP_DELAY). Keep these in sync if you retune the grasp.
STEP_DEG = 5.0      # max degrees a servo moves per step (smaller = slower/smoother)
STEP_DELAY = 0.15   # seconds paused between steps (bigger = slower)


def main():
    act = ArmActuator()
    angles = list(NEUTRAL) + [90.0] * (NUM_CH - len(NEUTRAL))  # CH7-16 default to mid
    last_ch = 0
    step = 5.0

    def apply(i, val):
        """Move CH(i+1) to val SLOWLY via the shared stepped_move (only this channel
        moves; the others have zero delta and are skipped)."""
        val = max(0.0, min(270.0, float(val)))
        target = list(angles)
        target[i] = val
        stepped_move(act, angles, target, STEP_DEG, STEP_DELAY)
        angles[i] = val
        print(f"  CH{i + 1} = {val:.1f}°")

    def print_pose():
        print("\n  current servo angles (CH1-7):")
        print("   ", [round(a, 1) for a in angles[:7]])
        print("    arm only (CH1-5):", [round(a, 1) for a in angles[:5]])
        if _kin is not None:
            try:
                tip = _kin.predict_tip(angles[:5])
                print(f"    model thinks tip is at (x,y,z) = "
                      f"{[round(v * 100, 1) for v in tip]} cm")
            except Exception as e:
                print(f"    (tip preview failed: {e})")
        print()

    print("=" * 60)
    print(" SERVO JOG & POSE CAPTURE")
    print(" type 'h' to home, '2 160' to set a servo, 'p' to print, 'q' to quit")
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
            # Cut PWM to every channel: duty 0 -> no pulse -> servos go limp.
            # Stops twitching, and nothing is held so power-cycling won't snap back.
            for i in range(6):
                act.kit.servo[i].angle = None
            print("  RELEASED all servos (no signal — arm is limp). "
                  "Set any channel to re-engage.")
        elif cmd == "h":
            # Home all 6 arm/gripper channels TOGETHER (not one channel at a time).
            target = list(angles)
            for i, a in enumerate(NEUTRAL):
                target[i] = a
            stepped_move(act, angles, target, STEP_DEG, STEP_DELAY)
            for i, a in enumerate(NEUTRAL):
                angles[i] = a
            print(f"  homed CH1-6 -> {[round(a, 1) for a in NEUTRAL]}")
            last_ch = 0
        elif cmd == "p":
            print_pose()
        elif cmd == "step" and len(parts) == 2:
            try:
                step = float(parts[1])
                print(f"  nudge step = {step}°")
            except ValueError:
                print("  usage: step <degrees>")
        elif cmd == "s":
            name = parts[1] if len(parts) > 1 else "unnamed"
            with open(POSE_FILE, "a", encoding="utf-8") as f:
                f.write(f"{name}\t{datetime.now():%Y-%m-%d %H:%M}\t"
                        f"{[round(a, 1) for a in angles]}\n")
            print(f"  saved '{name}' -> {POSE_FILE}")
        elif cmd.startswith(("+", "-")) and len(cmd) > 1 and cmd[1:].isdigit():
            # "+2" / "-3": nudge that channel by +/- step
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
                    print("  angle must be a number 0-270")
            else:
                print(f"  channel must be 1-{NUM_CH}")
        else:
            print("  commands: 'CH angle' | '+CH'/'-CH' | 'step N' | p | s name | h | q")

    print("done (servos still holding — press 'r' before 'q' to leave the arm limp).")


if __name__ == "__main__":
    main()
