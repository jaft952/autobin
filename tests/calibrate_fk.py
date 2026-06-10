"""
Forward-Kinematics calibration helper.

Goal: before trusting IK, make the MODEL's prediction match the REAL arm. This script
prints, for a set of known servo poses, where the model thinks the gripper tip is. You
move the real arm to the same servo angles, measure the tip with a ruler, and compare.

How to read it:
  * Frame origin = base of CH1, +Z points up. All positions printed in centimeters.
  * If the NEUTRAL pose isn't physically vertical / the right height -> fix zero offsets
    (JOINT_OFFSET_DEG) and rotation-axis directions FIRST, everything else depends on it.
  * In the single-joint sweeps, watch which way the REAL tip moves vs. the model. If a
    joint moves the opposite way, that joint's direction is wrong: flip its `rotation`
    axis in kinematics.py (e.g. [1,0,0] -> [-1,0,0]) OR toggle it in REVERSED_JOINTS —
    pick ONE place, never both (they cancel).

Run:  python tests/calibrate_fk.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.kinematics import ArmKinematics, REVERSED_JOINTS


def show(kin, label, servo):
    tip = kin.predict_tip(servo)
    tip_cm = [round(v * 100, 1) for v in tip]
    print(f"  {label:14s} servo={servo}  ->  model tip (x,y,z) cm = {tip_cm}")
    return tip_cm


def main():
    kin = ArmKinematics()
    print("=" * 72)
    print(" FK CALIBRATION  —  compare MODEL prediction vs. RULER measurement")
    print(f" current REVERSED_JOINTS = {sorted(REVERSED_JOINTS)}   (verify this set below!)")
    print("=" * 72)

    print("\n[1] NEUTRAL pose — real arm should stand straight up & centered:")
    show(kin, "neutral", [135, 135, 135, 135, 135])
    print("    EXPECT physically ~ (0, 0, 45.8) cm, vertical.")
    print("    If not vertical / wrong height -> fix zero offsets & axis directions first.")

    print("\n[2] Single-joint sweeps — move ONE joint, check the tip direction matches:")
    base = [135, 135, 135, 135, 135]
    for ch in range(1, 6):
        print(f"  -- CH{ch} --")
        for val in (90, 135, 180):
            servo = list(base)
            servo[ch - 1] = val
            show(kin, f"CH{ch}={val}", servo)
    print("\n  NOTE: CH5 is roll about the tool axis — it should NOT move the tip position.")
    print("        'Y reversed, X fine' usually points at CH1 (base yaw) reversed first.")

    print("\n[3] Interactive compare (optional). Enter a measured pose, or 'q' to quit.")
    print("    Format:  ch1 ch2 ch3 ch4 ch5 = mx my mz   (servo degrees = measured cm)")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() == "q":
            break
        if "=" not in line:
            print("    need an '=' separating servo angles from measured cm")
            continue
        try:
            left, right = line.split("=", 1)
            servo = [float(p) for p in left.split()]
            meas = [float(p) for p in right.split()]
            if len(servo) != 5 or len(meas) != 3:
                print("    need 5 servo angles and 3 measured cm values")
                continue
            model_cm = [round(v * 100, 1) for v in kin.predict_tip(servo)]
            delta = [round(meas[i] - model_cm[i], 1) for i in range(3)]
            print(f"    model={model_cm} cm   measured={meas} cm   delta(meas-model)={delta} cm")
        except ValueError as e:
            print(f"    parse error: {e}")


if __name__ == "__main__":
    main()
