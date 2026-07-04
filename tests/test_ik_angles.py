"""
tests/test_ik_angles.py

Pure-computation IK inspector — NO hardware, no I2C. Runs on Windows or the Pi.

Purpose: see EXACTLY what servo angle the IK computes for each servo, and — when
the solver says "unreachable" — see the closest pose it found and WHICH joint /
how far over its limit it went. This separates three different failures:

  * IK math wrong        -> analytic vs ikpy disagree, or model tip != target
  * joint-limit too tight-> a position-reaching pose exists but a joint is "OVER LIMIT"
  * calibration wrong    -> use the 'fk' command: feed your KNOWN-GOOD hardcoded
                            servo angles and compare the model tip to the ruler.

Usage:
    python tests/test_ik_angles.py                 # interactive
    python tests/test_ik_angles.py 0.1 0.2 0.1     # one-shot report for a target

Interactive commands:
    x y z              solve IK for this target and report every servo angle
    fk c1 c2 c3 c4 c5  model tip for these 5 servo commands (calibration check)
    down / free        toggle the gripper-DOWN constraint (default: free)
    q                  quit
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to cp1252, which can't encode the emoji below.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from src.arm.analytical_ik import (
    AnalyticalArmIK, _ik_to_servo, DOWN_PITCH_RAD, GRASP_TILTS_DEG,
)
from src.arm.kinematics import (
    ArmKinematics, joint_range_deg, IK_POSITION_TOLERANCE,
)

CH = ["CH1 base", "CH2 shoulder", "CH3 elbow", "CH4 wrist_pitch", "CH5 roll"]


def fmt_servo(servo):
    """Clean, plain-number servo line: '123.3  149.4  41.3  138.6  90.0'."""
    return "  ".join(f"{float(v):6.1f}" for v in servo)


def diagnose(analytic, target, grasp_down):
    """List every position-reaching pose the geometry allows (ignoring joint limits),
    pick the one that strains the joints least, and show per-joint usage so you can
    see if a real, physical pose is only being rejected by the limit/margin."""
    x, y, z = target
    if grasp_down:
        pitches = [DOWN_PITCH_RAD + np.radians(t) for t in GRASP_TILTS_DEG]
    else:
        pitches = [np.radians(p) for p in range(-180, 91, 10)]

    candidates = []  # (max_usage, angles)
    for theta1, r in analytic._yaw_branches(x, y):
        for phi4 in pitches:
            for (t2, t3, t4) in analytic._planar_solutions(r, z, phi4):
                angles = [0.0, theta1, t2, t3, t4, 0.0, 0.0]
                tip = analytic._fk(angles)
                if np.linalg.norm(tip - np.array([x, y, z])) > IK_POSITION_TOLERANCE:
                    continue  # this pose does not actually land on the target
                # Usage vs the ASYMMETRIC window: fraction of the room available
                # in the direction the joint actually swings.
                usage = []
                for i in range(1, 6):
                    deg = np.degrees(angles[i])
                    lo, hi = joint_range_deg(i)
                    room = hi if deg >= 0 else lo
                    usage.append(abs(deg / room) if abs(room) > 1e-9 else float("inf"))
                candidates.append((max(usage), angles, usage))

    if not candidates:
        print("  ✗ No position-reaching pose exists at ANY approach angle.")
        print("    -> target is outside the LINK GEOMETRY (too far / too close), not a limit issue.")
        return

    candidates.sort(key=lambda c: c[0])
    worst, angles, usage = candidates[0]
    servo = _ik_to_servo(angles)
    over = worst > 1.0
    print(f"  Closest pose that reaches the point — servo CH1-5:")
    print(f"    {fmt_servo(servo)}")
    print(f"  Each joint can only swing so far from its rest position; here is how")
    print(f"  much of that allowance each one needs (over 100% = the IK forbids it):")
    for i in range(1, 6):
        u = usage[i - 1] * 100
        flag = "  <-- TOO FAR, this joint blocks it" if u > 100 else ""
        print(f"    {CH[i-1]:<14} {u:4.0f}%{flag}")
    if over:
        print("  => A pose that REACHES the point exists, but a joint's allowed range")
        print("     stops the IK. If the real servo physically goes there, the limit is too tight.")


def report(analytic, kin, target, grasp_down):
    print("=" * 56)
    print(f"TARGET {target}   mode={'DOWN' if grasp_down else 'FREE'}")
    print("-" * 56)

    servo = analytic.solve(list(target), grasp_down=grasp_down)
    if servo is not None:
        tip = kin.predict_tip(servo)
        err = np.linalg.norm(np.array(tip) - np.array(target)) * 100
        print("  ✅ REACHABLE")
        print("  SERVO ANGLES to send   CH1    CH2    CH3    CH4    CH5")
        print(f"                       {fmt_servo(servo)}")
        landed = tuple(round(float(t), 3) for t in tip)
        print(f"  Model says gripper lands at {landed} m  (off target by {err:.1f} cm)")
    else:
        print("  ❌ The IK will NOT move here. Closest it can get, and why:")
        diagnose(analytic, target, grasp_down)
    print("=" * 56 + "\n")


def do_fk(kin, servo5):
    """Model tip for 5 hardcoded servo commands — the calibration comparison."""
    tip = kin.predict_tip(servo5)
    landed = tuple(round(float(t), 3) for t in tip)
    print(f"  servo {fmt_servo(servo5)}")
    print(f"  ->  model says the gripper is at {landed} m")
    print(f"  Compare that to where the REAL gripper actually is (ruler).\n")


def main():
    analytic = AnalyticalArmIK()
    kin = ArmKinematics()

    # One-shot mode: `python tests/test_ik_angles.py 0.1 0.2 0.1`
    if len(sys.argv) == 4:
        target = tuple(float(v) for v in sys.argv[1:4])
        report(analytic, kin, target, grasp_down=False)
        return

    print("Type a target 'x y z' to get the servo angles. Other commands:")
    print("  fk c1 c2 c3 c4 c5  -> where the model thinks those servo angles land")
    print("  down / free        -> gripper must point down, or don't care (default free)")
    print("  q                  -> quit")
    grasp_down = False

    while True:
        try:
            raw = input(f"\n[{'DOWN' if grasp_down else 'FREE'}] > ").strip().lower()
            if raw == 'q':
                break
            if raw == 'down':
                grasp_down = True
                continue
            if raw == 'free':
                grasp_down = False
                continue

            parts = raw.split()
            if parts and parts[0] == 'fk':
                if len(parts) != 6:
                    print("  usage: fk c1 c2 c3 c4 c5  (5 servo commands)")
                    continue
                do_fk(kin, [float(p) for p in parts[1:6]])
                continue

            if len(parts) == 3:
                target = tuple(float(p) for p in parts)
                report(analytic, kin, target, grasp_down)
            else:
                print("  enter 'x y z', or 'fk c1..c5', or down/free/q")
        except ValueError:
            print("  numbers only, e.g. '0.1 0.2 0.1' or 'fk 96.7 96.7 100 100 90'")
        except (KeyboardInterrupt, EOFError):
            break

    print("bye")


if __name__ == "__main__":
    main()
