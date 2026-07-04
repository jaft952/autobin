from __future__ import annotations
import numpy as np

from src.arm.kinematics import (
    SERVO_NEUTRAL_CMD,
    CMD_PER_DEG,
    SERVO_CMD_MAX,
    REVERSED_JOINTS,
    IK_POSITION_TOLERANCE,
    joint_range_rad,
)

# Planar geometry — MUST match the link translations in kinematics.py's chain.
# RULER-MEASURED on the real arm (2026-07-03), axis-center to axis-center.
# z=0 is the surface the 2.9cm mounting block sits on (chassis deck).
H_SHOULDER = 0.029 + 0.069      # A: deck -> CH2 (shoulder) axis = 9.8cm measured
L1 = 0.105                      # B: CH2 -> CH3 = 10.5cm measured
L2 = 0.128                      # C: CH3 -> CH4 = 12.8cm measured
L3 = 0.031 + 0.1555             # D: CH4 -> gripper tip = 18.65cm measured

DOWN_PITCH_RAD = -np.pi         # cumulative pitch for the gripper pointing straight down
UP_PITCH_RAD = 0.0              # cumulative pitch for the gripper pointing straight up
GRASP_TILTS_DEG = (0.0, 15.0, 30.0, 45.0)   # tilt ladder away from vertical to try
# Prefer pointing more straight-down, but accept tilt if it gives a much comfier pose.
TILT_PENALTY_PER_DEG = 0.004
_EPS = 1e-6


def _ik_to_servo(ik):
    """ik joint angles (radians, index 1..5) -> physical servo commands [CH1..CH5].
    Same mapping as kinematics.ArmKinematics (single source = the calibration constants)."""
    out = []
    for i in range(1, 6):
        direction = -1.0 if i in REVERSED_JOINTS else 1.0
        cmd = SERVO_NEUTRAL_CMD[i] + direction * np.degrees(ik[i]) * CMD_PER_DEG
        out.append(round(max(0.0, min(SERVO_CMD_MAX, cmd)), 2))
    return out


class AnalyticalArmIK:
    """Closed-form IK. solve() returns physical servo angles [CH1..CH5] or None."""

    def __init__(self):
        # Per-joint ASYMMETRIC angle windows (radians, index 1..5): the full servo
        # command range mapped around each neutral — see kinematics.joint_range_deg.
        ranges = [(0.0, 0.0)] + [joint_range_rad(i) for i in range(1, 6)]
        self.lo = [r[0] for r in ranges]
        self.hi = [r[1] for r in ranges]

    # ── public ────────────────────────────────────────────────────────────
    def solve(self, target_xyz, grasp_down: bool = True, approach: str = None):
        """
        grasp_down=True : gripper points down (tries straight-down, then small outward
                          tilts), picking the comfiest reachable pose.
        grasp_down=False: position only — sweep the approach angle and pick the comfiest
                          reachable pose (no orientation requirement).
        approach        : overrides grasp_down when given — "down", "up" (gripper
                          pointing up, approaching from below; tilt ladder tries both
                          sides of vertical), or "free".
        """
        x, y, z = (float(v) for v in target_xyz)

        if approach is None:
            approach = "down" if grasp_down else "free"
        if approach == "down":
            pitches = [(DOWN_PITCH_RAD + np.radians(t), t) for t in GRASP_TILTS_DEG]
        elif approach == "up":
            pitches = [(UP_PITCH_RAD + np.radians(s * t), t)
                       for t in GRASP_TILTS_DEG
                       for s in ((1.0,) if t == 0.0 else (1.0, -1.0))]
        elif approach == "free":
            pitches = [(np.radians(p), None) for p in range(-180, 91, 10)]
        else:
            raise ValueError(f"approach must be 'down', 'up' or 'free', got {approach!r}")

        best = None  # (cost, angles)
        for theta1, r in self._yaw_branches(x, y):
            if not (self.lo[1] - _EPS <= theta1 <= self.hi[1] + _EPS):
                continue
            for phi4, tilt in pitches:
                for theta2, theta3, theta4 in self._planar_solutions(r, z, phi4):
                    angles = [0.0, theta1, theta2, theta3, theta4, 0.0, 0.0]
                    usage = self._max_joint_usage(angles)
                    if usage > 1.0 + _EPS:
                        continue  # outside joint limits
                    cost = usage + (TILT_PENALTY_PER_DEG * tilt if tilt else 0.0)
                    if best is None or cost < best[0]:
                        best = (cost, angles)

        if best is None:
            return None

        angles = best[1]
        if np.linalg.norm(self._fk(angles) - np.array([x, y, z])) > IK_POSITION_TOLERANCE:
            return None
        return _ik_to_servo(angles)

    # ── geometry ──────────────────────────────────────────────────────────
    @staticmethod
    def _yaw_branches(x, y):
        """CH1 options: face the target directly (reach forward) or face away (reach
        'backward' with negative planar radius). Returns (theta1, signed_r) pairs."""
        psi = np.arctan2(y, x)
        R = np.hypot(x, y)
        fwd = AnalyticalArmIK._wrap(psi - np.pi / 2.0)
        bwd = AnalyticalArmIK._wrap(psi + np.pi / 2.0)
        return [(fwd, R), (bwd, -R)]

    def _planar_solutions(self, r, z, phi4):
        """Solve the planar 3-link arm for end point (r, z) with end pitch phi4.
        Returns up to two (theta2, theta3, theta4) tuples (elbow-down / elbow-up)."""
        # Wrist (CH4) position required so the L3 segment ends at (r, z) pointing along phi4.
        v_wrist = r + L3 * np.sin(phi4)
        u_wrist = z - L3 * np.cos(phi4)
        dv = v_wrist
        du = u_wrist - H_SHOULDER
        D2 = dv * dv + du * du
        D = np.sqrt(D2)
        if D > (L1 + L2) - _EPS or D < abs(L1 - L2) + _EPS:
            return []  # 2-link arm can't reach the wrist point

        cos3 = np.clip((D2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0)
        out = []
        for sign in (-1.0, 1.0):
            theta3 = sign * np.arccos(cos3)
            gamma2 = np.arctan2(du, dv) - np.arctan2(L2 * np.sin(theta3),
                                                     L1 + L2 * np.cos(theta3))
            theta2 = gamma2 - np.pi / 2.0          # convert (-sin,cos) frame to joint angle
            theta4 = phi4 - (theta2 + theta3)      # cumulative pitch must equal phi4
            out.append((self._wrap(theta2), self._wrap(theta3), self._wrap(theta4)))
        return out

    @staticmethod
    def _fk(angles):
        """Planar forward kinematics of [_, t1, t2, t3, t4, _, _] -> world (x, y, z)."""
        t1, t2, t3, t4 = angles[1], angles[2], angles[3], angles[4]
        p2, p3, p4 = t2, t2 + t3, t2 + t3 + t4
        v = -(L1 * np.sin(p2) + L2 * np.sin(p3) + L3 * np.sin(p4))
        u = H_SHOULDER + L1 * np.cos(p2) + L2 * np.cos(p3) + L3 * np.cos(p4)
        return np.array([v * -np.sin(t1), v * np.cos(t1), u])

    # ── helpers ───────────────────────────────────────────────────────────
    def _max_joint_usage(self, angles):
        """Worst per-joint usage over CH1..CH5 against the ASYMMETRIC windows:
        theta/hi when swinging positive, theta/lo when negative (both ratios are
        positive fractions of the available room in that direction).
        <=1 means within limits, lower = comfier."""
        worst = 0.0
        for i in range(1, 6):
            a = angles[i]
            if a >= 0.0:
                room = self.hi[i]
                if room <= _EPS:
                    if a > _EPS:
                        return float("inf")   # no positive travel at all
                    continue
                worst = max(worst, a / room)
            else:
                room = self.lo[i]
                if room >= -_EPS:
                    return float("inf")       # no negative travel at all
                worst = max(worst, a / room)
        return worst

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
