import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


# ── Servo calibration (YF-6125MG) ────────────────────────────────────────────
SERVO_CMD_MAX = 180.0      # servo command spans 0..180 (== actuation_range)
CMD_PER_DEG = 1.0          # command units per physical degree
SERVO_NEUTRAL_CMD = [0.0, 96.7, 96.7, 100.0, 100.0, 90.0, 80.0]  # index 0 = OriginLink, 6 = gripper
REVERSED_JOINTS = {1, 2, 4} 
IK_POSITION_TOLERANCE = 0.01  # meters


SEED_SHOULDER_RAD = np.radians(45)
SEED_ELBOW_RAD = np.radians(-60)
SEED_WRIST_RAD = np.radians(45)
# Safety margin off the servo's 0/180 command ends. The hardcoded poses the arm
# runs every day already command up to 180 (e.g. GRAB_ANGLES CH4), so 5 deg was
# over-conservative — it alone rejected near-workspace targets like (0,0.2,0)
# whose best pose needs a joint within ~3 deg of the command end.
JOINT_BOUND_MARGIN_DEG = 2.0


def joint_range_deg(i: int) -> tuple:
    """Usable IK-angle window (lo_deg, hi_deg) of joint i: the full 0..SERVO_CMD_MAX
    command window mapped through neutral/direction, minus a safety margin at each end.
    ASYMMETRIC because the neutrals aren't 90 — e.g. CH3 (neutral 100, not reversed)
    can swing -95..+75 deg, and CH4 (neutral 100, reversed) -75..+95. The old
    symmetric +/-min(n, 180-n) bound threw the extra ~15-20 deg away and rejected
    physically reachable targets."""
    n = SERVO_NEUTRAL_CMD[i]
    direction = -1.0 if i in REVERSED_JOINTS else 1.0
    a = direction * (JOINT_BOUND_MARGIN_DEG - n) / CMD_PER_DEG
    b = direction * ((SERVO_CMD_MAX - JOINT_BOUND_MARGIN_DEG) - n) / CMD_PER_DEG
    return (a, b) if a <= b else (b, a)


def joint_range_rad(i: int) -> tuple:
    lo, hi = joint_range_deg(i)
    return (np.radians(lo), np.radians(hi))

GRIPPER_DOWN = [0.0, 0.0, -1.0]
GRIPPER_UP   = [0.0, 0.0, 1.0]   # approach from below, gripper pointing up
# Horizontal approach (grip a standing tin at its middle). Not a fixed vector —
# the actual direction depends on the target azimuth, so this is a sentinel that
# calculate_servo_angles/GraspPlanner resolve per target.
GRIPPER_LEVEL = "level"
APPROACH_TILTS_DEG = (0.0, 25.0, 45.0)
ORIENTATION_WARN_DEG = 20.0


class ArmKinematics:
    """
    Inverse Kinematics engine for the custom 5-DOF YF-6125MG manipulator.
    Coordinates are in meters. Zero-position is straight up along Z-axis.
    """
    def __init__(self):
        # Link lengths RULER-MEASURED on the real arm (2026-07-03, axis-center
        # to axis-center, arm vertical). z=0 = the surface the 2.9cm mounting
        # block sits on (chassis deck): A = 2.9 block + 6.9 = 9.8 to CH2 axis.
        # The original guessed lengths had B/C's long and short segments swapped
        # and D far too short — tip landed ~6cm too far forward and ~13cm too low.
        self.chain = Chain(name="yf_6125mg_arm", links=[
            OriginLink(),

            URDFLink(
                name="base_rotation",           # CH1: Yaw (2.9cm mounting block)
                origin_translation=[0, 0, 0.029],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                bounds=joint_range_rad(1),
            ),

            URDFLink(
                name="shoulder",                # CH2: Pitch (A: deck->axis 9.8 total)
                origin_translation=[0, 0, 0.069],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                bounds=joint_range_rad(2),
            ),

            URDFLink(
                name="elbow",                   # CH3: Pitch (B: CH2->CH3 = 10.5)
                origin_translation=[0, 0, 0.105],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                bounds=joint_range_rad(3),
            ),

            URDFLink(
                name="wrist_pitch",             # CH4: Pitch (C: CH3->CH4 = 12.8)
                origin_translation=[0, 0, 0.128],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                bounds=joint_range_rad(4),
            ),

            URDFLink(
                name="wrist_rotate",            # CH5: Roll (inside segment D)
                origin_translation=[0, 0, 0.031],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                bounds=joint_range_rad(5),
            ),

            # Tool Center Point (TCP) = gripper tip. Completes segment D:
            # CH4 axis -> closed gripper tip = 18.65cm measured (3.1 + 15.55).
            URDFLink(
                name="gripper_tcp",             # CH6 Handled elsewhere (Claw state)
                origin_translation=[0, 0, 0.1555],
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
        x, y, _z = target_xyz
        phi = np.arctan2(y, x)
        theta1 = phi - family * (np.pi / 2.0)
        # Wrap to [-pi, pi] then clamp into CH1's reachable range.
        theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
        lo1, hi1 = joint_range_rad(1)
        theta1 = float(np.clip(theta1, lo1, hi1))

        return [
            0.0,                # OriginLink 
            theta1,             # CH1: aimed at target azimuth
            SEED_SHOULDER_RAD,  # CH2
            SEED_ELBOW_RAD,     # CH3
            SEED_WRIST_RAD,     # CH4
            0.0,                # CH5 (roll — no effect on position)
            0.0,                # TCP 
        ]

    # ------------------------------------------------------------------ #
    # Angle <-> servo mapping (single source of truth for both directions)
    # ------------------------------------------------------------------ #
    def _ik_to_servo(self, ik_angles_rad) -> list:
        servo_angles_deg = []
        for i in range(1, 6):  # CH1 .. CH5
            direction = -1.0 if i in REVERSED_JOINTS else 1.0
            cmd = SERVO_NEUTRAL_CMD[i] + direction * np.degrees(ik_angles_rad[i]) * CMD_PER_DEG
            cmd = max(0.0, min(SERVO_CMD_MAX, cmd))
            servo_angles_deg.append(round(cmd, 2))
        return servo_angles_deg

    def _servo_to_ik(self, servo_deg: list) -> list:
        ik = [0.0]  # OriginLink
        for idx, i in enumerate(range(1, 6)):
            direction = -1.0 if i in REVERSED_JOINTS else 1.0
            ik_deg = direction * (servo_deg[idx] - SERVO_NEUTRAL_CMD[i]) / CMD_PER_DEG
            ik.append(np.radians(ik_deg))
        ik.append(0.0)  # TCP
        return ik

    def predict_tip(self, servo_deg: list) -> list:
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
        target = np.asarray(target_xyz, dtype=float)

        # Resolve the LEVEL sentinel into a horizontal unit vector pointing at
        # the target azimuth (falls back to +Y if the target is on the axis).
        is_level = tool_direction is GRIPPER_LEVEL
        if is_level:
            r = float(np.hypot(target[0], target[1]))
            tool_direction = ([float(target[0] / r), float(target[1] / r), 0.0]
                              if r > 1e-9 else [0.0, 1.0, 0.0])

        # Candidate approach directions: the tilt ladder for the default grasp mode,
        # exactly what the caller asked for otherwise.
        if tool_direction is GRIPPER_DOWN:
            directions = self._down_directions(target_xyz)
        elif tool_direction is not None:
            directions = [(None, tool_direction)]
        else:
            directions = [(None, None)]

        # Seed priority: analytic closed-form solution (if one exists, the optimizer
        # starts AT a valid pose and converges immediately) -> azimuth-aimed branch
        # -> last good pose (warm start) -> mirror branch.
        seeds = []
        try:
            # Lazy import: analytical_ik imports from this module, so a top-level
            # import here would be circular.
            from src.arm.analytical_ik import AnalyticalArmIK
            if tool_direction is GRIPPER_DOWN:
                _approach = "down"
            elif tool_direction is GRIPPER_UP:
                _approach = "up"
            elif is_level:
                _approach = "level"
            else:
                _approach = "free"
            servo = AnalyticalArmIK().solve(list(target_xyz), approach=_approach)
            if servo is not None:
                seeds.append(self._servo_to_ik(servo))
        except Exception:
            pass  # analytic seeding is best-effort; numeric seeds below still apply
        seeds.append(self._make_seed(target_xyz, family=1))
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
