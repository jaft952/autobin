"""
Servo calibration tool — find why "commanded degrees != physical degrees".

The arm moves across the full 0-270 command range (no dead zone), so the problem
is SCALE: a commanded change of N degrees may produce fewer physical degrees,
because actuation_range / pulse-width don't match the servo's true travel.

You do NOT need a protractor. "Vertical" and "horizontal" are exact 90° references,
so we use the arm itself to measure the command->physical scale.

Menu:
  s  SCALE CHECK (recommended) — measure command:physical ratio with a right angle,
     and get the recommended actuation_range. Do this on CH2.
  r  ACTUATION-RANGE experiment — try 180 / 200 / 240 / 270 / custom and re-check.
  p  PULSE-WIDTH experiment — try different microsecond ranges.
  l  LIMIT FINDER — step outward until the servo stops (saturation), answer f/s/n.
  q  quit

Run:  python tests/test_servo_travel.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.actuators.pca9685_driver import (
    ArmActuator,
    SERVO_MIN_PULSE_US,
    SERVO_MAX_PULSE_US,
    SERVO_RANGE_DEG,
)

ACTUATION_PRESETS = [180, 200, 220, 240, 270]
PULSE_PRESETS = [(500, 2500), (600, 2400), (750, 2250), (450, 2550), (400, 2600)]
NEUTRAL = [135.0] * 5


def main():
    act = ArmActuator()
    print("=" * 66)
    print(" SERVO CALIBRATION TOOL")
    print(f" startup: actuation_range={SERVO_RANGE_DEG}°, pulse={SERVO_MIN_PULSE_US}-{SERVO_MAX_PULSE_US}us")
    print("=" * 66)

    try:
        ch = int(input("\nWhich channel to test? (1-6) [2 is best for scale]: ").strip())
    except ValueError:
        print("invalid channel")
        return
    if not (1 <= ch <= 6):
        print("channel must be 1-6")
        return
    idx = ch - 1
    # Track the actuation_range currently applied to this channel.
    state = {"range": float(SERVO_RANGE_DEG)}

    def home():
        for i, a in enumerate(NEUTRAL):
            act.kit.servo[i].angle = a

    def write(angle):
        angle = max(0.0, min(state["range"], float(angle)))
        act.kit.servo[idx].angle = angle
        print(f"  -> commanded CH{ch} = {angle}°  (max={state['range']:.0f})")
        return angle

    def scale_check():
        print("\n--- SCALE CHECK (right-angle method) ---")
        home()
        print("All servos -> 135. The arm should now stand VERTICAL (straight up).")
        input("Confirm it's vertical, then press Enter... ")
        print("\nNow bring CH%d so the arm segment is exactly HORIZONTAL (level with the"
              " table). Type a command number to move it; type 'h' when it's level." % ch)
        print("(vertical was command 135; just type values like 90, 70, 50 ... and watch)")
        last = 135.0
        while True:
            s = input("  CH command / 'h'=now horizontal / 'q'=abort: ").strip().lower()
            if s == "q":
                home()
                return
            if s == "h":
                delta = abs(last - 135.0)
                if delta < 5:
                    print("  that's barely off vertical — move it to truly horizontal first.")
                    continue
                scale = 90.0 / delta
                recommended = round(state["range"] * 90.0 / delta)
                print("\n  ===== RESULT =====")
                print(f"  vertical@135 -> horizontal@{last:.0f}: command moved {delta:.0f}°"
                      f" to make a real 90°.")
                print(f"  command:physical scale = {scale:.2f}"
                      f"  ({'≈1.0 = already correct!' if 0.9 <= scale <= 1.1 else 'NOT 1:1'})")
                print(f"  -> recommended actuation_range ≈ {recommended}° "
                      f"(servo's true travel; currently set to {state['range']:.0f}).")
                print("  Tell me this number for ALL three joints (or once if same servo model).")
                home()
                return
            try:
                last = write(float(s))
            except ValueError:
                print("  type a number, or 'h' / 'q'.")

    def set_range(r):
        state["range"] = float(r)
        act.kit.servo[idx].actuation_range = float(r)
        print(f"  CH{ch} actuation_range set to {r}° (neutral is now {r/2:.0f}).")

    def range_experiment():
        print("\nactuation_range presets:", ", ".join(str(x) for x in ACTUATION_PRESETS))
        raw = input("pick a value (or custom number): ").strip()
        try:
            set_range(float(raw))
        except ValueError:
            print("  not a number."); return
        print("Now re-run the scale check with this range to see if it's 1:1.")
        scale_check()

    def pulse_experiment():
        print("\npulse presets:")
        for i, (lo, hi) in enumerate(PULSE_PRESETS, 1):
            print(f"  {i}) {lo}-{hi} us")
        raw = input("pick number or 'lo hi': ").strip().split()
        try:
            lo, hi = (PULSE_PRESETS[int(raw[0]) - 1] if len(raw) == 1
                      else (int(raw[0]), int(raw[1])))
            act.kit.servo[idx].set_pulse_width_range(lo, hi)
            print(f"  CH{ch} pulse range -> {lo}-{hi} us")
        except (ValueError, IndexError):
            print("  bad input."); return
        scale_check()

    def limit_finder():
        print("\n--- LIMIT FINDER ---  answer: f=full step  s=smaller  n=no move")
        for label, steps in (("UP", range(150, int(state["range"]) + 1, 15)),
                             ("DOWN", range(120, -1, -15))):
            home()
            input(f"centered — Enter to step {label}... ")
            last_full, prev = 135, 135
            for a in steps:
                write(a)
                ans = input("   f/s/n? ").strip().lower()
                if ans == "f":
                    last_full = a
                elif ans in ("s", "n"):
                    print(f"   -> saturates between {prev} and {a}")
                    break
                prev = a
            print(f"   {label} fully-responding to ~{last_full}")
        home()

    actions = {"s": scale_check, "r": range_experiment,
               "p": pulse_experiment, "l": limit_finder}
    while True:
        choice = input("\nmenu [s=scale  r=range  p=pulse  l=limits  q=quit]: ").strip().lower()
        if choice == "q":
            home()
            break
        action = actions.get(choice)
        if action:
            action()
        else:
            print("  pick s / r / p / l / q")


if __name__ == "__main__":
    main()
