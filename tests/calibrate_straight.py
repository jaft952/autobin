"""
tests/calibrate_straight.py

Straightness-trim finder for one direction of travel.

    python tests/calibrate_straight.py --reverse     # the one that needs it
    python tests/calibrate_straight.py               # forward, to re-check

!!! ROBOT ON THE FLOOR, ~2 m OF CLEAR SPACE, HAND ON CTRL+C !!!

WHY: motor_b_*_trim scales the stronger motor down so the base runs
straight. The forward pair is measured (0.75); motor_b_backward_trim is
still 0.95, which is nowhere near the same mismatch -- so every reverse
burst veers. Layer 3's overshoot retreat is a reverse burst, and it steers
the base off the spot the arc grid solved for, once per pulse.

HOW: drives a short straight burst, you say which way it veered, it moves
the trim and repeats. Stop when two bursts in a row run straight, then put
the printed value into src/motion/calibration.py.

Burst only -- no key latches continuous motion.
"""
import argparse
import dataclasses
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics
from src.hardware.actuators.pwm_driver import PWMActuator

TRIM_STEP = 0.05
TRIM_MIN, TRIM_MAX = 0.40, 1.00


def _ask_veer() -> str:
    while True:
        answer = input("    veered [l]eft / [r]ight / [s]traight / [q]uit? ").strip().lower()
        if answer[:1] in ("l", "r", "s", "q"):
            return answer[0]


def main():
    ap = argparse.ArgumentParser(description="find motor_b_*_trim for one direction")
    ap.add_argument("--reverse", action="store_true", help="calibrate backward travel")
    ap.add_argument("--speed", type=float, default=20.0, help="duty, 0..100")
    ap.add_argument("--seconds", type=float, default=1.5, help="burst length")
    args = ap.parse_args()

    name = "motor_b_backward_trim" if args.reverse else "motor_b_forward_trim"
    base = MotionCalibration()
    trim = getattr(base, name)

    print(__doc__.split("WHY:")[0])
    print(f"calibrating {name}, starting at {trim}")
    print(f"burst: {'backward' if args.reverse else 'forward'} "
          f"at duty {args.speed} for {args.seconds}s\n")

    act = None
    straight_runs = 0
    try:
        while straight_runs < 2:
            cal = dataclasses.replace(base, **{name: trim})
            kin = DifferentialKinematics(cal)
            # Rebuild the actuator so it holds this trim, and close the old
            # one first -- two drivers on the same pins fight each other.
            if act is not None:
                act.close()
            act = PWMActuator(calibration=cal)

            input(f"\n=== {name} = {trim:.2f} ===\n    Enter to run...")
            cmd = kin.backward(speed=args.speed) if args.reverse else \
                kin.forward(speed=args.speed)
            act.apply(cmd)
            time.sleep(args.seconds)
            act.stop()

            veer = _ask_veer()
            if veer == "q":
                break
            if veer == "s":
                straight_runs += 1
                print(f"    straight ({straight_runs}/2)")
                continue

            straight_runs = 0
            # Veering toward a side means that side is travelling less far,
            # so the OTHER motor is over-driven and its trim must come down.
            # Channel B is the trimmed one; which way that lands depends on
            # the wiring, so follow what the robot does, not the labels.
            trim += TRIM_STEP if veer == "l" else -TRIM_STEP
            trim = max(TRIM_MIN, min(TRIM_MAX, trim))
            print(f"    -> trying {trim:.2f}")

        if straight_runs >= 2:
            print(f"\nSET IN src/motion/calibration.py:\n    {name}: float = {trim:.2f}")
    except KeyboardInterrupt:
        print("\naborted")
    finally:
        if act is not None:
            act.stop()
            act.close()
        print("motors stopped, GPIO released")


if __name__ == "__main__":
    main()
