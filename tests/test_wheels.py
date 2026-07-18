"""
tests/test_wheels.py

Wheel direction / speed / brake checker for the ZK-BM1 motor driver.
Run this FIRST after any motor-driver or wiring change — the calibration
flags (invert_left / invert_right / swap_left_right in
src/motion/calibration.py) were tuned for the OLD wiring and may be wrong
for the new one.

    python tests/test_wheels.py            # interactive, Enter between steps
    python tests/test_wheels.py --speed 0.4

!!! PUT THE ROBOT ON A BOX SO THE WHEELS ARE OFF THE GROUND !!!

Fix table (edit src/motion/calibration.py after observing):
  FORWARD spins BOTH wheels backward   -> flip invert_left AND invert_right
  FORWARD spins ONE wheel backward     -> flip that side's invert_* (note:
                                          with swap_left_right=True the
                                          labels are crossed — trust what
                                          you SEE, flip, re-run)
  TURN LEFT actually turns right       -> flip swap_left_right
  everything reversed AND mirrored     -> flip all three, re-run
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics
from src.hardware.actuators.pwm_driver import PWMActuator

RUN_S = 1.5


def scaled(cmd, factor):
    cmd.left_speed *= factor
    cmd.right_speed *= factor
    return cmd


def step(title, expect):
    input(f"\n=== {title} ===\n    expect: {expect}\n    Enter to run...")


def main():
    ap = argparse.ArgumentParser(description="ZK-BM1 wheel direction checker")
    ap.add_argument("--speed", type=float, default=0.45, help="0..1 duty factor")
    args = ap.parse_args()

    cal = MotionCalibration()
    kin = DifferentialKinematics(cal)
    act = PWMActuator(calibration=cal)
    print("wheels driver up (ZK-BM1, sign-magnitude PWM). Ctrl+C aborts safely.")
    print(f"flags now: invert_left={cal.invert_left} invert_right={cal.invert_right} "
          f"swap_left_right={cal.swap_left_right}")

    try:
        step("FORWARD", "BOTH wheels spin the robot-forward direction")
        act.apply(scaled(kin.forward(), args.speed))
        time.sleep(RUN_S)
        act.stop()

        step("BACKWARD", "both wheels spin backward")
        act.apply(scaled(kin.backward(), args.speed))
        time.sleep(RUN_S)
        act.stop()

        step("TURN LEFT (pivot)", "LEFT wheel backward, RIGHT wheel forward")
        act.apply(scaled(kin.turn_left(), args.speed))
        time.sleep(RUN_S)
        act.stop()

        step("TURN RIGHT (pivot)", "left forward, right backward")
        act.apply(scaled(kin.turn_right(), args.speed))
        time.sleep(RUN_S)
        act.stop()

        step("BRAKE from speed", "wheels stop almost INSTANTLY (vs rolling out)")
        act.apply(scaled(kin.forward(), args.speed))
        time.sleep(RUN_S)
        act.brake()
        time.sleep(1.0)

        print("\n=== HOLD test: wheels are now BRAKED ===")
        input("    try turning a wheel BY HAND — it should resist. Enter to release...")
        act.stop()
        input("    now COASTING — the same wheel should spin freely. Enter to finish...")

        print("\nAll steps done. If any direction was wrong, edit the flags in "
              "src/motion/calibration.py (table in this file's header) and re-run.")
    except KeyboardInterrupt:
        print("\naborted.")
    finally:
        act.close()


if __name__ == "__main__":
    main()
