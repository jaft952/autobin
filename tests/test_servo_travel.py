"""
Servo travel calibration — confirms whether commanded degrees == physical degrees.

This is the decisive test for the "arm barely bends / z stays ~constant" problem. If a
servo, commanded from 0° to 270°, does NOT physically sweep a full 270°, the pulse-width
range is wrong and the arm can't reach the poses IK asks for.

You need a protractor / angle gauge on the joint you test.

Run:  python tests/test_servo_travel.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.actuators.pca9685_driver import (
    ArmActuator,
    SERVO_MIN_PULSE_US,
    SERVO_MAX_PULSE_US,
)


def main():
    act = ArmActuator()
    print("=" * 64)
    print(" SERVO TRAVEL TEST")
    print(f" pulse-width range in use: {SERVO_MIN_PULSE_US}-{SERVO_MAX_PULSE_US} us")
    print(" Put a protractor on the joint. Command vs physical angle should match.")
    print("=" * 64)

    try:
        ch = int(input("\nWhich channel to test? (1-6): ").strip())
    except ValueError:
        print("invalid channel")
        return
    if not (1 <= ch <= 6):
        print("channel must be 1-6")
        return
    idx = ch - 1

    def write(angle):
        act.kit.servo[idx].angle = max(0.0, min(270.0, float(angle)))
        print(f"  -> commanded CH{ch} = {angle}°")

    print("\nStep through commands; read the protractor at each and note the PHYSICAL angle.")
    print("(Start near the middle to stay safe; do the 0/270 extremes last.)")
    sequence = [135, 90, 135, 180, 135]
    for a in sequence:
        write(a)
        input("   read protractor, then Enter for next... ")

    if input("\nTest the 0° / 270° extremes too? (may hit mechanical stops) [y/N]: ").strip().lower() == "y":
        for a in (0, 270, 135):
            write(a)
            input("   read protractor, then Enter... ")

    print("\nHOW TO READ THE RESULT:")
    print(" * Command 90 -> 180 should physically move EXACTLY 90°.")
    print("   - moves LESS than 90° -> servo under-travels -> WIDEN the pulse range")
    print(f"     (lower {SERVO_MIN_PULSE_US} / raise {SERVO_MAX_PULSE_US} in pca9685_driver.py).")
    print("   - moves MORE than 90° -> NARROW the pulse range.")
    print(" * Command 0 -> 270 should sweep a full 270°.")
    print(" Tell me the physical angles you read and I'll set the exact pulse numbers.")


if __name__ == "__main__":
    main()
