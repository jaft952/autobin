"""
src/arm/analytical_ik.py

Closed-form inverse kinematics tailored to this 5-DOF arm. Replaces the numerical
ikpy solver for normal use (ikpy ArmKinematics is kept as a backup). Why custom:
  * the arm is simple enough for an exact geometric solution (no convergence failures);
  * we get ALL solution branches, so we can pick the most COMFORTABLE one (joints far
    from their limits) -> less strain / sag;
  * the gripper approach angle is an explicit input, so "point down" (or any tilt) is
    a parameter, not something we coax out of an optimizer.

Structure (matches the ikpy chain + calibration in kinematics.py exactly):
  * CH1 (yaw, Z) sets the azimuth: theta1 = atan2(y, x) - 90deg.
  * CH2/CH3/CH4 (pitch, X) form a PLANAR 3-link arm in the vertical plane, reaching
    the planar point (r = hypot(x,y), z) with a chosen end pitch phi4 (the gripper
    approach angle). Lengths: shoulder at height H_SHOULDER, links L1, L2, then L3 to
    the TCP. Solved by the standard wrist-partition + 2-link law of cosines (two elbow
    branches).
  * CH5 (roll, Z) does not affect position -> left at 0.

Angle convention (same as the ikpy model): at model-zero all angles are 0 and the arm
points straight up (+Z). A segment whose cumulative pitch (sum of CH2..CHi) is phi points
in the planar direction (-sin phi, cos phi) = (horizontal, vertical). Straight DOWN is
phi4 = -180deg.
"""

from __future__ import annotations
import numpy as np

from src.arm.kinematics import (
    SERVO_NEUTRAL_CMD,
    CMD_PER_DEG,
    SERVO_CMD_MAX,
    REVERSED_JOINTS,
    IK_POSITION_TOLERANCE,
    joint_bound_rad,
)

# Planar geometry — MUST match the link translations in kinematics.py's chain.
H_SHOULDER = 0.042 + 0.105      # CH2 (shoulder) axis height above the base origin
L1 = 0.1275                     # CH2 -> CH3
L2 = 0.070                      # CH3 -> CH4
L3 = 0.031 + 0.083              # CH4 -> TCP (wrist_rotate + gripper, collinear)

DOWN_PITCH_RAD = -np.pi         # cumulative pitch for the gripper pointing straight down
GRASP_TILTS_DEG = (0.0, 15.0, 30.0, 45.0)   # outward tilt from straight-down to try
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
        self.bounds = [0.0] + [joint_bound_rad(i) for i in range(1, 6)]  # radians, index 1..5

    # ── public ────────────────────────────────────────────────────────────
    def solve(self, target_xyz, grasp_down: bool = True):
        """
        grasp_down=True : gripper points down (tries straight-down, then small outward
                          tilts), picking the comfiest reachable pose.
        grasp_down=False: position only — sweep the approach angle and pick the comfiest
                          reachable pose (no orientation requirement).
        """
        x, y, z = (float(v) for v in target_xyz)

        if grasp_down:
            pitches = [(DOWN_PITCH_RAD + np.radians(t), t) for t in GRASP_TILTS_DEG]
        else:
            pitches = [(np.radians(p), None) for p in range(-180, 91, 10)]

        best = None  # (cost, angles)
        for theta1, r in self._yaw_branches(x, y):
            if abs(theta1) > self.bounds[1] + _EPS:
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
        """max |theta_i| / bound_i over CH1..CH5; <=1 means within limits, lower = comfier."""
        worst = 0.0
        for i in range(1, 6):
            b = self.bounds[i]
            if b <= _EPS:           # CH5 (roll) has a tiny modelled range; ignore it
                continue
            worst = max(worst, abs(angles[i]) / b)
        return worst

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
