import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


# ── Servo calibration (YF-6125MG) ────────────────────────────────────────────
# These servos are labelled "270°" but actually travel only ~180° with the 500-2500us
# pulse range, so the system uses actuation_range = 180 (see SERVO_RANGE_DEG in
# pca9685_driver.py — the two MUST match). With actuation_range = 180, one servo command
# unit equals one physical degree (1:1):
SERVO_CMD_MAX = 180.0      # servo command spans 0..180 (== actuation_range)
CMD_PER_DEG = 1.0          # command units per physical degree
# Servo command at each joint's MODEL-ZERO ("straight up") pose, read directly with
# tests/servo_jog.py at actuation_range=180 (CH1-5 = [96.7, 96.7, 100.0, 100.0, 90.0]).
# CH2/CH3/CH4 (pitch) determine reach and are solid; CH1 (yaw azimuth-zero) and CH5 (roll)
# aren't pinned down by verticality, so re-check those if the x/y direction looks rotated.
SERVO_NEUTRAL_CMD = [0.0, 96.7, 96.7, 100.0, 100.0, 90.0, 80.0]  # index 0 = OriginLink, 6 = gripper

# Joints whose physical servo is mounted in the reversed direction (negative physical
# motion for positive model angle). Determined per-joint via tests/calibrate_fk.py:
# CH1/CH2/CH4 are reversed; CH3 matches the model; CH5 is roll (no position effect).
REVERSED_JOINTS = {1, 2, 4}  # 1-based joint indices

# IK is considered converged only if the solution's forward kinematics lands within
# this distance of the requested target. A genuinely converged solve is sub-mm; 1 cm
# cleanly separates "reached it" from "solver railed against a joint limit".
IK_POSITION_TOLERANCE = 0.01  # meters

# Mid-range starting pose for shoulder/elbow/wrist used when seeding the solver.
# Deliberately NOT on any joint bound, so the local optimizer has room to converge.
SEED_SHOULDER_RAD = np.radians(45)
SEED_ELBOW_RAD = np.radians(-60)
SEED_WRIST_RAD = np.radians(45)


# Keep IK solutions this far (deg) off each joint's physical limit. Poses sitting exactly
# on a limit are mechanically marginal (the servo hunts/strains under gravity -> trembling).
JOINT_BOUND_MARGIN_DEG = 5.0


def joint_half_range_deg(i: int) -> float:
    """Usable half-range (degrees) of joint i: how far its command can swing from neutral
    within the 0..SERVO_CMD_MAX window (1:1 scale), minus a safety margin off the limit."""
    n = SERVO_NEUTRAL_CMD[i]
    physical = min(n, SERVO_CMD_MAX - n) / CMD_PER_DEG
    return max(5.0, physical - JOINT_BOUND_MARGIN_DEG)


def joint_bound_rad(i: int) -> float:
    return np.radians(joint_half_range_deg(i))

# Default tool approach direction: gripper pointing straight DOWN (world -Z).
# Position-only IK on this redundant 5-DOF arm picks arbitrary wrist poses (gripper may
# face up/sideways -> cannot grasp, and its length skews the measured tip). Constraining
# the tool axis to point down makes the pose unique and graspable. Pass tool_direction=None
# to fall back to pure position IK.
GRIPPER_DOWN = [0.0, 0.0, -1.0]

# When GRIPPER_DOWN is requested but straight-down is unreachable, retry with the
# approach tilted outward (toward the target azimuth) by these angles, in order.
# A vertical gripper costs its full 11.4cm of length in height budget; tilting ~45°
# recovers a lot of horizontal reach while still grasping top-down-ish.
APPROACH_TILTS_DEG = (0.0, 25.0, 45.0)
# If the achieved tool axis deviates from the requested direction by more than this, warn.
ORIENTATION_WARN_DEG = 20.0


class ArmKinematics:
    """
    Inverse Kinematics engine for the custom 5-DOF YF-6125MG manipulator.
    Coordinates are in meters. Zero-position is straight up along Z-axis.
    """
    def __init__(self):
        self.chain = Chain(name="yf_6125mg_arm", links=[
            OriginLink(),

            URDFLink(
                name="base_rotation",           # CH1: 0°-270° (Yaw)
                origin_translation=[0, 0, 0.042],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                bounds=(-joint_bound_rad(1), joint_bound_rad(1)),
            ),

            URDFLink(
                name="shoulder",                # CH2: 0°-270° (Pitch)
                origin_translation=[0, 0, 0.105],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                bounds=(-joint_bound_rad(2), joint_bound_rad(2)),
            ),

            URDFLink(
                name="elbow",                   # CH3: 0°-270° (Pitch)
                origin_translation=[0, 0, 0.1275],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                bounds=(-joint_bound_rad(3), joint_bound_rad(3)),
            ),

            URDFLink(
                name="wrist_pitch",             # CH4: 0°-270° (Pitch)
                origin_translation=[0, 0, 0.070],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                bounds=(-joint_bound_rad(4), joint_bound_rad(4)),
            ),

            URDFLink(
                name="wrist_rotate",            # CH5: 0°-270° (Roll)
                origin_translation=[0, 0, 0.031],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                bounds=(-joint_bound_rad(5), joint_bound_rad(5)),
            ),

            # Tool Center Point (TCP). It represents the tip of the gripper.
            URDFLink(
                name="gripper_tcp",             # CH6 Handled elsewhere (Claw state)
                origin_translation=[0, 0, 0.083],
                origin_orientation=[0, 0, 0],
                rotation=None,
                joint_type="fixed",
            ),
        ])

        # Last *converged* IK solution, reused as a warm-start seed for the next call.
        # Only updated on success, so a failed (railed) solve never poisons the next seed.
        self._last_angles = None
        self._last_solution_ok = False

    # ------------------------------------------------------------------ #
    # Seeding
    # ------------------------------------------------------------------ #
    def _make_seed(self, target_xyz: list, family: int = 1) -> list:
        """
        Build a full IK seed vector aimed at the target.

        The base joint (CH1) is the only DOF that produces horizontal azimuth, so a
        bad CH1 seed is what previously railed the solver against its ±135° bound. We
        point CH1 straight at the target azimuth instead of starting near a bound.

        family = +1 / -1 selects which of the two mirror solution branches to seed
        (arm reaching "forward" vs "backward"), letting the caller retry the other
        branch if the first fails.
        """
        x, y, _z = target_xyz
        phi = np.arctan2(y, x)
        theta1 = phi - family * (np.pi / 2.0)
        # Wrap to [-pi, pi] then clamp into CH1's reachable range.
        theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
        theta1 = float(np.clip(theta1, -joint_bound_rad(1), joint_bound_rad(1)))

        return [
            0.0,                # OriginLink (fixed)
            theta1,             # CH1: aimed at target azimuth
            SEED_SHOULDER_RAD,  # CH2
            SEED_ELBOW_RAD,     # CH3
            SEED_WRIST_RAD,     # CH4
            0.0,                # CH5 (roll — no effect on position)
            0.0,                # TCP (fixed)
        ]

    # ------------------------------------------------------------------ #
    # Angle <-> servo mapping (single source of truth for both directions)
    # ------------------------------------------------------------------ #
    def _ik_to_servo(self, ik_angles_rad) -> list:
        """
        Map ikpy's internal joint angles (radians, 0 = model-zero pose) to physical
        servo commands. Applies per-joint neutral, the measured command-per-degree
        scale, and the reversed-mount direction:  servo = neutral ± ik_deg * scale.
        """
        servo_angles_deg = []
        for i in range(1, 6):  # CH1 .. CH5
            direction = -1.0 if i in REVERSED_JOINTS else 1.0
            cmd = SERVO_NEUTRAL_CMD[i] + direction * np.degrees(ik_angles_rad[i]) * CMD_PER_DEG
            cmd = max(0.0, min(SERVO_CMD_MAX, cmd))
            servo_angles_deg.append(round(cmd, 2))
        return servo_angles_deg

    def _servo_to_ik(self, servo_deg: list) -> list:
        """
        Inverse of _ik_to_servo: physical servo commands -> ikpy 7-vector (radians).
        Used by the calibration tooling to predict where a given servo pose lands.
        (Hardware clamping at 0/270 is not invertible; assumes values within range.)
        """
        ik = [0.0]  # OriginLink
        for idx, i in enumerate(range(1, 6)):
            direction = -1.0 if i in REVERSED_JOINTS else 1.0
            ik_deg = direction * (servo_deg[idx] - SERVO_NEUTRAL_CMD[i]) / CMD_PER_DEG
            ik.append(np.radians(ik_deg))
        ik.append(0.0)  # TCP
        return ik

    def predict_tip(self, servo_deg: list) -> list:
        """
        Calibration helper: given 5 physical servo angles (CH1-CH5), return the
        model's predicted TCP position [x, y, z] in meters. Compare against a ruler
        to validate/repair link lengths, zero offsets and rotation-axis directions.
        """
        ik = self._servo_to_ik(servo_deg)
        tip = self.chain.forward_kinematics(ik)[:3, 3]
        return [round(float(v), 4) for v in tip]

    # ------------------------------------------------------------------ #
    # Inverse kinematics
    # ------------------------------------------------------------------ #
    def tool_axis(self, ik_angles) -> np.ndarray:
        """World-space direction the gripper points (its local +Z) for an IK solution."""
        return np.asarray(self.chain.forward_kinematics(ik_angles))[:3, 2]

    def reset_warm_start(self):
        """Forget the stored warm-start pose. Call after the arm is moved outside IK
        (e.g. servo-level homing), so the next solve doesn't seed from a stale pose."""
        self._last_angles = None
        self._last_solution_ok = False

    def _down_directions(self, target_xyz):
        """Candidate approach directions for grasping: straight down first, then tilted
        outward toward the target azimuth per APPROACH_TILTS_DEG. Returns a list of
        (tilt_deg, direction_vector) pairs."""
        x, y, _z = target_xyz
        phi = np.arctan2(y, x)
        out = []
        for tilt_deg in APPROACH_TILTS_DEG:
            t = np.radians(tilt_deg)
            out.append((tilt_deg, [float(np.sin(t) * np.cos(phi)),
                                   float(np.sin(t) * np.sin(phi)),
                                   float(-np.cos(t))]))
        return out

    def calculate_servo_angles(self, target_xyz: list, tool_direction=GRIPPER_DOWN,
                               orientation_mode="Z") -> list | None:
        """
        Solve IK for target_xyz (meters) and map the result to physical 0°-270°
        servo angles [CH1..CH5].

        By default the gripper is constrained to point DOWN (tool_direction=GRIPPER_DOWN,
        orientation_mode="Z"), which removes the wrist redundancy so the pose is unique and
        graspable. If straight-down is unreachable, the approach is retried tilted outward
        per APPROACH_TILTS_DEG (still top-down enough to grasp, much larger workspace).
        Pass tool_direction=None for pure position IK.

        Robustness: try several seeds per direction and accept the first whose forward
        kinematics lands within IK_POSITION_TOLERANCE of the target.

        Returns the 5 servo angles, or None if nothing converges (target unreachable /
        outside the joint-limited workspace). None means "do not move the arm".
        """
        target = np.asarray(target_xyz, dtype=float)

        # Candidate approach directions: the tilt ladder for the default grasp mode,
        # exactly what the caller asked for otherwise.
        if tool_direction is GRIPPER_DOWN:
            directions = self._down_directions(target_xyz)
        elif tool_direction is not None:
            directions = [(None, tool_direction)]
        else:
            directions = [(None, None)]

        # Seed priority: azimuth-aimed branch -> last good pose (warm start) -> mirror branch.
        seeds = [self._make_seed(target_xyz, family=1)]
        if self._last_solution_ok and self._last_angles is not None:
            seeds.append(list(self._last_angles))
        seeds.append(self._make_seed(target_xyz, family=-1))

        best_sol = None
        best_err = float("inf")
        solved_tilt = None
        solved_direction = None

        for tilt_deg, direction in directions:
            for seed in seeds:
                try:
                    if direction is not None:
                        sol = self.chain.inverse_kinematics(
                            target_position=target_xyz,
                            target_orientation=direction,
                            orientation_mode=orientation_mode,
                            initial_position=seed,
                            max_iter=1000,
                        )
                    else:
                        sol = self.chain.inverse_kinematics(
                            target_position=target_xyz,
                            initial_position=seed,
                            max_iter=1000,
                        )
                except Exception as e:
                    print(f"[Kinematics] IK solver exception: {e}")
                    continue

                tip = self.chain.forward_kinematics(sol)[:3, 3]
                err = float(np.linalg.norm(tip - target))
                if err < best_err:
                    best_err, best_sol = err, sol
                if err <= IK_POSITION_TOLERANCE:
                    solved_tilt = tilt_deg
                    solved_direction = direction
                    break  # good enough, stop trying seeds
            if best_err <= IK_POSITION_TOLERANCE:
                break  # stop trying more-tilted directions

        if solved_tilt is not None and solved_tilt > 0:
            print(f"[Kinematics] straight-down unreachable; using approach tilted "
                  f"{solved_tilt:.0f}° from vertical toward the target.")

        # --- Convergence check: refuse to fake success ---
        if best_sol is None or best_err > IK_POSITION_TOLERANCE:
            print(
                f"[Kinematics] No convergent IK solution for {target_xyz} "
                f"(best residual = {best_err * 100:.2f} cm > {IK_POSITION_TOLERANCE * 100:.1f} cm). "
                f"Likely unreachable / outside joint-limited workspace."
            )
            self._last_solution_ok = False
            return None

        # Warn if the gripper could not actually reach the orientation that was solved for.
        if solved_direction is not None:
            axis = self.tool_axis(best_sol)
            want = np.asarray(solved_direction, dtype=float)
            want = want / np.linalg.norm(want)
            tilt = np.degrees(np.arccos(np.clip(float(np.dot(axis, want)), -1.0, 1.0)))
            if tilt > ORIENTATION_WARN_DEG:
                print(f"[Kinematics] note: gripper orientation off by {tilt:.0f}° "
                      f"from requested (tool axis {np.round(axis, 2)}).")

        # Success: remember the pose for warm-starting the next call.
        self._last_angles = best_sol.tolist()
        self._last_solution_ok = True
        return self._ik_to_servo(best_sol)
