"""
Servo travel calibration — confirms whether commanded degrees == physical degrees,
and lets you experiment with different pulse-width ranges live.

Two distinct problems this can reveal (they can coexist):
  * OFFSET — every step moves the RIGHT amount, but the whole range is rotated
    (e.g. "the arm is shifted left overall"). Cause: servo horn mounted a few spline
    teeth off / zero offset. Fix: remount the horn, or compensate in JOINT_OFFSET_DEG.
    Pulse width will NOT fix this.
  * SCALE — command 0..270 sweeps LESS than 270° physically (steps are short).
    Cause: pulse-width range too narrow for this servo, or the servo's real travel
    is simply less than 270°. Fix: widen pulses (this script lets you try), or accept
    the real travel and narrow the IK joint bounds.

Quick discrimination: command 135 -> 225. Physically exactly 90° = OFFSET only.
Physically ~60° = SCALE problem.

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

# Candidate pulse ranges (min_us, max_us) to try. Wider range = more travel per command,
# but pushing past the servo's electrical limits makes it buzz/stall at the extremes —
# if that happens, back off immediately (enter 135) and try a narrower range.
PULSE_PRESETS = [
    (500, 2500),   # typical 270° spec (current default)
    (600, 2400),
    (750, 2250),   # adafruit_servokit default (typical 180° spec)
    (450, 2550),
    (400, 2600),
]


def choose_pulse_range():
    print("\nPulse-width presets:")
    for i, (lo, hi) in enumerate(PULSE_PRESETS, start=1):
        mark = "  <- current default" if (lo, hi) == (SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US) else ""
        print(f"  {i}) {lo}-{hi} us{mark}")
    print("  or type a custom range like: 550 2450")
    raw = input("pick preset number or custom range [Enter = keep current]: ").strip()
    if not raw:
        return SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US
    parts = raw.split()
    try:
        if len(parts) == 1:
            return PULSE_PRESETS[int(parts[0]) - 1]
        if len(parts) == 2:
            lo, hi = int(parts[0]), int(parts[1])
            if 300 <= lo < hi <= 3000:
                return lo, hi
    except (ValueError, IndexError):
        pass
    print("  didn't understand that — keeping current range.")
    return SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US


def main():
    act = ArmActuator()
    print("=" * 64)
    print(" SERVO TRAVEL TEST")
    print(f" startup pulse-width range: {SERVO_MIN_PULSE_US}-{SERVO_MAX_PULSE_US} us")
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

    def find_limits():
        """Step outward from center until the servo stops responding (saturation).
        No protractor needed — you only answer whether it MOVED."""
        print("\n[limit finder] I'll step 15° at a time. After each step answer:")
        print("   f = moved a FULL step   s = moved but SMALLER   n = did NOT move")
        results = {}
        for direction, steps in (("UP", range(150, 271, 15)), ("DOWN", range(120, -1, -15))):
            write(135)
            input(f"\ncentered at 135 — Enter to start stepping {direction}... ")
            last_full = 135
            prev = 135
            for a in steps:
                write(a)
                ans = input("   f / s / n ? ").strip().lower()
                if ans == "f":
                    last_full = a
                elif ans in ("s", "n"):
                    print(f"   -> saturation between command {prev} and {a}")
                    break
                prev = a
            results[direction] = last_full
        write(135)
        up, down = results.get("UP", 135), results.get("DOWN", 135)
        print("\n=== LIMIT FINDER RESULT ===")
        print(f"  CH{ch} responds fully for commands ~{down} to ~{up}"
              f"  (= {up - down}° of command range)")
        print(f"  -> report these two numbers to set the IK joint bounds correctly.")

    if input("\nRun the LIMIT FINDER first? (no protractor needed) [Y/n]: ").strip().lower() != "n":
        find_limits()
        if input("\nContinue to pulse-range experiments? [y/N]: ").strip().lower() != "y":
            return

    while True:
        lo, hi = choose_pulse_range()
        act.kit.servo[idx].set_pulse_width_range(lo, hi)
        print(f"\n### CH{ch} now using pulse range {lo}-{hi} us ###")
        write(135)
        input("centered at 135 — mark/note the physical direction, then Enter... ")

        print("\n[scale check] 135 -> 225 should physically move EXACTLY 90°:")
        write(225)
        input("   measure the physical movement, then Enter... ")
        write(135)

        print("\n[full sweep] watch each step (45° expected per step except as labeled):")
        for a, expect in ((90, "135->90: 45°"), (180, "90->180: 90°"), (225, "180->225: 45°"),
                          (250, "225->250: 25°"), (270, "250->270: 20°"), (135, "back to center")):
            write(a)
            input(f"   expect {expect}; check angle + buzzing/stall/collision, then Enter... ")

        if input("\nTest the 0° extreme too? (may hit mechanical stops) [y/N]: ").strip().lower() == "y":
            for a in (0, 135):
                write(a)
                input("   read protractor, then Enter... ")

        again = input("\nTry ANOTHER pulse range on this channel? [y/N]: ").strip().lower()
        if again != "y":
            break

    print("\nHOW TO READ THE RESULT:")
    print(" * 135->225 moves exactly 90° but the whole range points the wrong way")
    print("   -> OFFSET problem: remount the servo horn / adjust zero. Pulses won't help.")
    print(" * steps consistently SHORT -> try a WIDER pulse range (rerun and pick one).")
    print(" * steps correct in 90-180 but short only near 225-270 -> servo/linkage limit:")
    print("   note the angle where it stops; we narrow the IK bounds to match.")
    print(" * found a range where steps are exact? -> tell me the numbers and I'll set")
    print("   SERVO_MIN/MAX_PULSE_US in pca9685_driver.py permanently.")


if __name__ == "__main__":
    main()
