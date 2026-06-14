"""
Forward-Kinematics calibration helper — makes the MODEL match the REAL arm.

This tool MOVES the real arm (on the Pi) and tells you where the MODEL thinks the
gripper tip is, so you can compare directions and distances against a ruler.

WHY: IK now converges in math, but the physical joint directions/zero points may not
match the model (symptom: "Y reversed", arm extends when it should fold). This tool
isolates each joint so we can find which ones are reversed.

COORDINATE FRAME (decide once and keep it):
  * +Z = up.
  * Pick a "front" for the robot. +Y = away from you (forward), -Y = toward you.
  * +X = to your right.

Run:  python tests/calibrate_fk.py
Commands at the prompt:
  g                 run the GUIDED single-joint test (recommended — do this first)
  n                 move to NEUTRAL (model-zero), arm should stand straight up
  a b c d e         move CH1..CH5 to these 5 angles (e.g. 135 95 135 135 135)
  cN v              change only channel N to angle v (e.g. c2 95)
  = x y z           record your ruler-measured tip (cm) and print delta vs model
  q                 quit
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.kinematics import ArmKinematics, REVERSED_JOINTS, SERVO_NEUTRAL_CMD

try:
    from src.hardware.actuators.pca9685_driver import ArmActuator
    _ACT_ERR = None
except Exception as e:  # pragma: no cover - depends on hardware libs
    ArmActuator = None
    _ACT_ERR = e

NEUTRAL = [SERVO_NEUTRAL_CMD[i] for i in range(1, 6)]  # model-zero pose (actuation_range=180)


class Calibrator:
    def __init__(self):
        self.kin = ArmKinematics()
        self.act = ArmActuator() if ArmActuator is not None else None
        if self.act is None:
            print(f"[note] actuator unavailable ({_ACT_ERR}); running in PREDICT-ONLY mode (no real movement).")
        self.last_servo = list(NEUTRAL)
        self.last_tip = self._tip(NEUTRAL)

    def _tip(self, servo):
        return [round(v * 100, 1) for v in self.kin.predict_tip(servo)]

    def move(self, servo):
        servo = [max(0.0, min(270.0, float(x))) for x in servo]
        tip = self._tip(servo)
        if self.act is not None:
            self.act.set_arm_angles(servo)
        self.last_servo, self.last_tip = servo, tip
        print(f"  -> CH1-5 = {servo}")
        print(f"     MODEL predicts tip (x, y, z) cm = {tip}")
        return tip

    @staticmethod
    def _dir_words(neutral_tip, tip):
        dx, dy, dz = (tip[i] - neutral_tip[i] for i in range(3))
        parts = []
        if abs(dx) >= 0.5:
            parts.append("+X(right)" if dx > 0 else "-X(left)")
        if abs(dy) >= 0.5:
            parts.append("+Y(away/forward)" if dy > 0 else "-Y(toward you)")
        if abs(dz) >= 0.5:
            parts.append("up" if dz > 0 else "down")
        return ", ".join(parts) if parts else "(barely moved)"

    def guided(self):
        print("\n" + "=" * 64)
        print(" GUIDED CALIBRATION — watch the REAL arm after each move.")
        print(" For each joint: does the real tip move the SAME way the model says,")
        print(" or the OPPOSITE? Write it down — that tells us the REVERSED set.")
        print("=" * 64)
        nt = self.move(NEUTRAL)
        input("\n[neutral] Is the real arm STANDING STRAIGHT UP & centered? (look, then Enter) ")

        # Move each tested joint -40° from its neutral command and observe direction.
        def pose_with(ch_idx, delta):
            p = list(NEUTRAL)
            p[ch_idx] = p[ch_idx] + delta
            return p

        for name, ch in (("CH2 (shoulder)", 2), ("CH3 (elbow)", 3), ("CH4 (wristpitch)", 4)):
            print(f"\n-- {name}: moving its servo {NEUTRAL[ch-1]:.0f} -> {NEUTRAL[ch-1]-40:.0f} --")
            tip = self.move(pose_with(ch - 1, -40))
            print(f"   MODEL says the tip moves: {self._dir_words(nt, tip)}")
            input("   Watch the REAL arm — note SAME or OPPOSITE, then Enter. ")

        # CH1 (yaw) only shows direction when the arm is tilted, so pre-tilt with CH2.
        print("\n-- CH1 (base yaw): first tilt arm forward, then rotate base --")
        base_tilt = pose_with(1, -40)  # CH2 tilted
        nt1 = self.move(base_tilt)
        input("   (arm now tilted) press Enter to rotate CH1 -40° ")
        tip = self.move([NEUTRAL[0] - 40] + base_tilt[1:])
        print(f"   MODEL says the tip swings: {self._dir_words(nt1, tip)}")
        input("   Watch the REAL arm — note SAME or OPPOSITE, then Enter. ")

        self.move(NEUTRAL)
        print("\nDone. Report for each joint whether real == model (SAME) or OPPOSITE.")
        print("Rule: every joint that is OPPOSITE must be TOGGLED in REVERSED_JOINTS.")

    def repl(self):
        print(f"\ncurrent REVERSED_JOINTS = {sorted(REVERSED_JOINTS)}")
        print("type 'g' for the guided test, or 'q' to quit. (see file header for all commands)")
        while True:
            try:
                line = input("\ncal> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() == "q":
                break
            low = line.lower()
            try:
                if low == "g":
                    self.guided()
                elif low == "n":
                    self.move(NEUTRAL)
                elif low.startswith("c") and len(line.split()) == 2 and line[1].isdigit():
                    ch = int(line.split()[0][1:])
                    val = float(line.split()[1])
                    servo = list(self.last_servo)
                    servo[ch - 1] = val
                    self.move(servo)
                elif line.startswith("="):
                    meas = [float(p) for p in line[1:].split()]
                    if len(meas) != 3:
                        print("   need 3 numbers: = x y z (cm)")
                        continue
                    delta = [round(meas[i] - self.last_tip[i], 1) for i in range(3)]
                    print(f"   model={self.last_tip} cm  measured={meas} cm  delta(meas-model)={delta} cm")
                else:
                    nums = [float(p) for p in line.split()]
                    if len(nums) != 5:
                        print("   give 5 angles (CH1..CH5), or a command (g/n/cN v/= x y z/q)")
                        continue
                    self.move(nums)
            except (ValueError, IndexError) as e:
                print(f"   parse error: {e}")


if __name__ == "__main__":
    Calibrator().repl()
