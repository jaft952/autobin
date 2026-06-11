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

    print("\n*** UPPER ZONE 180-270 — IMPORTANT for CH2/CH4! ***")
    print("IK solutions for forward grasping poses always land in this zone for the")
    print("reversed joints, and it was NOT covered by the basic sweep above.")
    print("At each step ALSO check: does the bracket hit the frame / does the servo")
    print("buzz or stall before reaching the angle?")
    if input("Test the 180-270 zone now? [Y/n]: ").strip().lower() != "n":
        for a in (180, 225, 250, 270, 135):
            write(a)
            input("   read protractor + check for collision/stall, then Enter... ")

    if input("\nTest the 0° extreme too? (may hit mechanical stops) [y/N]: ").strip().lower() == "y":
        for a in (0, 135):
            write(a)
            input("   read protractor, then Enter... ")

    print("\nHOW TO READ THE RESULT:")
    print(" * Command 90 -> 180 should physically move EXACTLY 90°.")
    print(" * Command 180 -> 225 -> 270 should each step EXACTLY 45°.")
    print("   - steps in 90-180 correct but SHORT in 180-270 -> servo can't do the top")
    print("     of its range (or the bracket collides) -> report which angle it stops at.")
    print("   - ALL steps proportionally short -> WIDEN the pulse range")
    print(f"     (lower {SERVO_MIN_PULSE_US} / raise {SERVO_MAX_PULSE_US} in pca9685_driver.py).")
    print(" Tell me the physical angles you read and I'll set the exact numbers.")


if __name__ == "__main__":
    main()
