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

import numpy as np

from src.arm.analytical_ik import (
    AnalyticalArmIK, _ik_to_servo, DOWN_PITCH_RAD, GRASP_TILTS_DEG,
)
from src.arm.kinematics import (
    ArmKinematics, GRIPPER_DOWN, SERVO_NEUTRAL_CMD, SERVO_CMD_MAX,
    joint_half_range_deg, IK_POSITION_TOLERANCE,
)

CH = ["CH1 base", "CH2 shoulder", "CH3 elbow", "CH4 wrist_pitch", "CH5 roll"]


def print_bounds():
    print("Joint limits currently enforced by the IK (from SERVO_NEUTRAL_CMD, minus 5° margin):")
    for i in range(1, 6):
        n = SERVO_NEUTRAL_CMD[i]
        half = joint_half_range_deg(i)
        # Physical one-sided room in servo units, ignoring the symmetric IK bound.
        room_lo, room_hi = n - 0.0, SERVO_CMD_MAX - n
        print(f"  {CH[i-1]:<14} neutral={n:6.1f}  IK bound=±{half:5.1f}°   "
              f"servo room: {room_lo:.0f}° down / {room_hi:.0f}° up")
    print()


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
                usage = [abs(np.degrees(angles[i])) / joint_half_range_deg(i) for i in range(1, 6)]
                candidates.append((max(usage), angles, usage))

    if not candidates:
        print("  ✗ No position-reaching pose exists at ANY approach angle.")
        print("    -> target is outside the LINK GEOMETRY (too far / too close), not a limit issue.")
        return

    candidates.sort(key=lambda c: c[0])
    worst, angles, usage = candidates[0]
    servo = _ik_to_servo(angles)
    over = worst > 1.0
    print(f"  closest position-reaching pose ({'OVER joint limits' if over else 'within limits'}):")
    print(f"    servo CH1-5 = {servo}")
    for i in range(1, 6):
        deg = np.degrees(angles[i])
        u = usage[i - 1] * 100
        flag = "  <-- OVER LIMIT" if u > 100 else ""
        print(f"    {CH[i-1]:<14} model {deg:7.1f}°   ({u:4.0f}% of ±{joint_half_range_deg(i):.0f}°){flag}")
    if over:
        print("    => a pose that REACHES the point exists, but the IK bound rejects it.")
        print("       If the real servo can physically go there, the bound/margin is too tight.")


def report(analytic, kin, target, grasp_down):
    print("=" * 64)
    print(f"TARGET {target}   mode={'DOWN' if grasp_down else 'FREE'}")
    print("-" * 64)

    # 1) What the analytic solver officially returns.
    servo = analytic.solve(list(target), grasp_down=grasp_down)
    if servo is None:
        print("[analytic] returns None (rejected).")
    else:
        tip = kin.predict_tip(servo)
        err = np.linalg.norm(np.array(tip) - np.array(target)) * 100
        print(f"[analytic] servo CH1-5 = {servo}")
        print(f"[analytic] model tip   = {tip}   residual {err:.2f} cm")

    # 2) ikpy backup solver, for cross-check.
    try:
        servo2 = kin.calculate_servo_angles(list(target),
                                            tool_direction=GRIPPER_DOWN if grasp_down else None)
        print(f"[ikpy]     servo CH1-5 = {servo2}")
    except Exception as e:
        print(f"[ikpy]     error: {e}")

    # 3) The diagnostic — closest reaching pose + per-joint usage.
    print("-" * 64)
    diagnose(analytic, target, grasp_down)
    print("=" * 64 + "\n")


def do_fk(kin, servo5):
    """Model tip for 5 hardcoded servo commands — the calibration comparison."""
    tip = kin.predict_tip(servo5)
    print(f"  servo {servo5}  ->  model tip {tip} (meters)")
    print(f"  compare that to where the REAL arm's gripper actually is with these angles.\n")


def main():
    analytic = AnalyticalArmIK()
    kin = ArmKinematics()

    # One-shot mode: `python tests/test_ik_angles.py 0.1 0.2 0.1`
    if len(sys.argv) == 4:
        target = tuple(float(v) for v in sys.argv[1:4])
        print_bounds()
        report(analytic, kin, target, grasp_down=False)
        return

    print_bounds()
    print("Commands: 'x y z' | 'fk c1 c2 c3 c4 c5' | 'down' | 'free' | 'q'")
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
