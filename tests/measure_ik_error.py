"""
tests/measure_ik_error.py

Map the IK error empirically: command several targets with the CURRENT inverse
kinematics, measure where the gripper REALLY ends up, and see whether the error
(real - target) is a CONSTANT offset or grows with the pose.

  * constant offset  -> just correct the target by that offset; no recalibration.
  * varies with pose -> joint direction/gain wrong or gravity sag; needs calibrate_fk.

ALL NUMBERS ARE IN CENTIMETRES here (easier to eyeball / ruler-measure).

Run on the Pi:  python tests/measure_ik_error.py

Commands:
  x y z         command this target (cm). Arm moves; then go measure the tip.
  = x y z       enter the ruler-measured tip (cm) for the target you just sent.
  table         show all recorded (target, measured, error) rows + is it constant?
  down / free   gripper must point down (grasp mode) / position-only. Default: down.
  q             quit
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.arm.analytical_ik import AnalyticalArmIK

try:
    from src.hardware.actuators.pca9685_driver import ArmActuator
    _ACT_ERR = None
except Exception as e:  # hardware libs may be absent off the Pi
    ArmActuator = None
    _ACT_ERR = e


def fmt(vec):
    return "(" + ", ".join(f"{v:+6.1f}" for v in vec) + ")"


class ErrorMapper:
    def __init__(self):
        self.ik = AnalyticalArmIK()
        self.act = ArmActuator() if ArmActuator is not None else None
        if self.act is None:
            print(f"[note] actuator unavailable ({_ACT_ERR}); PREDICT-ONLY, arm will NOT move.")
        self.grasp_down = True
        self.last_target_cm = None   # cm
        self.rows = []               # (target_cm, measured_cm, error_cm)

    def command(self, target_cm):
        target_m = [v / 100.0 for v in target_cm]
        servo = self.ik.solve(target_m, grasp_down=self.grasp_down)
        if servo is None:
            print(f"  ✗ IK cannot reach {fmt(target_cm)} cm in "
                  f"{'DOWN' if self.grasp_down else 'FREE'} mode — pick another point.")
            self.last_target_cm = None
            return
        servo = [round(float(s), 1) for s in servo]
        print(f"  target {fmt(target_cm)} cm  ->  servo CH1-5 = {servo}")
        if self.act is not None:
            self.act.set_arm_angles(servo)
            print("  arm moving. Now measure the real gripper tip, then:  = x y z   (cm)")
        else:
            print("  (no hardware) would send those servo angles.")
        self.last_target_cm = target_cm

    def measure(self, measured_cm):
        if self.last_target_cm is None:
            print("  send a target first (x y z), then measure.")
            return
        err = [measured_cm[i] - self.last_target_cm[i] for i in range(3)]
        self.rows.append((self.last_target_cm, measured_cm, err))
        print(f"  target {fmt(self.last_target_cm)}  measured {fmt(measured_cm)}  "
              f"ERROR(meas-target) {fmt(err)} cm")
        self.last_target_cm = None

    def table(self):
        if not self.rows:
            print("  no measurements yet.")
            return
        print("\n  target (cm)            measured (cm)          error (cm)")
        print("  " + "-" * 58)
        for t, m, e in self.rows:
            print(f"  {fmt(t):<22} {fmt(m):<22} {fmt(e)}")
        errs = np.array([e for _, _, e in self.rows])
        mean = errs.mean(axis=0)
        spread = errs.max(axis=0) - errs.min(axis=0)
        print("  " + "-" * 58)
        print(f"  mean error   {fmt(mean)} cm")
        print(f"  spread(max-min per axis) {fmt(spread)} cm")
        if np.all(spread <= 2.0):
            print("  => error is ~CONSTANT. Fix: aim at (target - mean_error).")
            print(f"     i.e. subtract {fmt(mean)} cm from every target before IK.")
        else:
            print("  => error VARIES with pose (spread > 2cm on some axis).")
            print("     Not a fixed offset -> joint direction/gain or sag. Use calibrate_fk.py.")
        print()

    def repl(self):
        print(f"\nmode = {'DOWN' if self.grasp_down else 'FREE'}. Type 'x y z' (cm) to start.")
        while True:
            try:
                line = input(f"\n[{'DOWN' if self.grasp_down else 'FREE'}] measure> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line == "q":
                break
            if line == "down":
                self.grasp_down = True
                continue
            if line == "free":
                self.grasp_down = False
                continue
            if line == "table":
                self.table()
                continue
            try:
                if line.startswith("="):
                    nums = [float(p) for p in line[1:].split()]
                    if len(nums) != 3:
                        print("  need 3 numbers: = x y z (cm)")
                        continue
                    self.measure(nums)
                else:
                    nums = [float(p) for p in line.split()]
                    if len(nums) != 3:
                        print("  give 'x y z' (cm), '= x y z', 'table', 'down/free', or 'q'")
                        continue
                    self.command(nums)
            except ValueError:
                print("  numbers only, e.g. '0 20 10' or '= 1 20 8'")

        self.table()
        print("bye")


if __name__ == "__main__":
    ErrorMapper().repl()
