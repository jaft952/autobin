"""
Software-level verification for the IK fixes (改动一). Hardware-independent: it only
exercises ArmKinematics (ikpy), never the servos, so it runs anywhere ikpy is installed
(Raspberry Pi or Windows). For the physical model-vs-reality calibration use
tests/calibrate_fk.py on the real arm instead.

Run:  python tests/test_ik_convergence.py
Exit code 0 = all hard checks passed.
"""
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.arm.kinematics import ArmKinematics, IK_POSITION_TOLERANCE, SERVO_NEUTRAL_CMD

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def fk_pos(kin, servo):
    """Forward-kinematics position (meters) of a 5-servo pose, via the model."""
    return np.array(kin.chain.forward_kinematics(kin._servo_to_ik(servo))[:3, 3], dtype=float)


def main():
    kin = ArmKinematics()

    print("\n[A] predict_tip / FK sanity")
    neutral_servo = [SERVO_NEUTRAL_CMD[i] for i in range(1, 6)]  # model-zero pose commands
    tip = kin.predict_tip(neutral_servo)
    # 0.5175 = H_SHOULDER 0.098 + L1 0.105 + L2 0.128 + L3 0.1865 (measured 2026-07-03)
    check("neutral tip ~ (0,0,0.5175) m", abs(tip[0]) < 1e-3 and abs(tip[1]) < 1e-3 and abs(tip[2] - 0.5175) < 2e-3, str(tip))

    print("\n[B] azimuth seed (_make_seed) points CH1 at the target")
    t1 = math.degrees(kin._make_seed([0.0, 0.1, 0.1], family=1)[1])   # +Y  -> expect ~0
    t2 = math.degrees(kin._make_seed([0.1, 0.1, 0.1], family=1)[1])   # 45° -> expect ~-45
    check("seed CH1 for +Y ~ 0 deg", abs(t1 - 0.0) < 1e-6, f"{t1:.2f}")
    check("seed CH1 for (10,10) ~ -45 deg", abs(t2 + 45.0) < 1e-6, f"{t2:.2f}")

    print("\n[C] servo<->ik mapping round-trips")
    ik = [0.0, math.radians(30), math.radians(-40), math.radians(50), math.radians(-20), math.radians(10), 0.0]
    rt = kin._servo_to_ik(kin._ik_to_servo(ik))
    err = max(abs(math.degrees(a - b)) for a, b in zip(ik, rt))
    check("ik->servo->ik error < 0.01 deg", err < 0.01, f"{err:.6f} deg")

    print("\n[D] IK convergence + honest reporting")
    # Comfortable mid-workspace target MUST converge.
    target = [0.20, 0.0, 0.20]
    servo = kin.calculate_servo_angles(target)
    if servo is None:
        check("comfortable target (0.20,0,0.20) converges", False, "got None")
    else:
        res = float(np.linalg.norm(fk_pos(kin, servo) - np.array(target)))
        check("comfortable target (0.20,0,0.20) converges", res <= IK_POSITION_TOLERANCE, f"residual={res*100:.2f} cm; servo={servo}")

    # Warm-start: a second call right after a success must still work.
    servo2 = kin.calculate_servo_angles([0.20, 0.05, 0.20])
    check("warm-started 2nd call returns a solution", servo2 is not None, str(servo2))

    print("\n[E] previously-broken low/close targets (informational — may be unreachable)")
    for t in ([0.0, 0.1, 0.1], [0.1, 0.1, 0.1]):
        s = kin.calculate_servo_angles(t)
        if s is None:
            print(f"  [INFO] {t}: honestly reported UNREACHABLE (no fake success).")
        else:
            res = float(np.linalg.norm(fk_pos(kin, s) - np.array(t)))
            print(f"  [INFO] {t}: converged, residual={res*100:.2f} cm, servo={s}")
            # If it returns a solution, that solution must be genuinely within tolerance.
            check(f"{t} returned solution is within tolerance", res <= IK_POSITION_TOLERANCE, f"residual={res*100:.2f} cm")

    print("\n[F] gripper points DOWN by default (orientation constraint)")
    servo = kin.calculate_servo_angles([0.20, 0.0, 0.20])  # default tool_direction = GRIPPER_DOWN
    if servo is None:
        check("comfortable target keeps gripper down", False, "got None")
    else:
        axis = kin.tool_axis(kin._servo_to_ik(servo))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(axis, [0, 0, -1]))))))
        check("gripper tool axis ~down (tilt < 25 deg)", tilt < 25.0, f"tilt={tilt:.1f} deg, axis={[round(float(v),2) for v in axis]}")

    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: all hard checks PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
