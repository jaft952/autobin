import math

from src.arm.kinematics import ArmKinematics, GRIPPER_DOWN, GRIPPER_UP, GRIPPER_LEVEL
from src.arm.analytical_ik import AnalyticalArmIK
from src.hardware.actuators.pca9685_driver import (
    ArmActuator, stepped_move, save_last_pose, load_last_pose,
    clamp_channel_angle,
)



HOME_ANGLES     = [100.0, 30.0, 0.0, 150.0, 90.0]   # user-set 2026-07-20
BIN_DROP_ANGLES = [96.7, 96.7, 100.0, 20.0, 90.0]
GRAB_ANGLES     = [101.0, 106.0, 40.0, 180.0, 80.0]

# MG996R gear gripper, safe window 0..55 enforced by CHANNEL_ANGLE_LIMITS.
# Keep in sync with tests/test_arc_grasp.py.
GRIPPER_OPEN = 50.0
GRIPPER_CLOSED = 4.0

# Arc-grasp sequence: calibrated poses assume this exact order and pacing.
ARC_LIFT_ARM      = [96.7, 96.7, 100.0, 100.0, 90.0]
# Grab order per tin pose: CH1 first, CH5 roll early, last channel descends
# onto the tin (upright: CH2 shoulder; lying: CH3 elbow).
ARC_GRAB_ORDER = {
    "upright": [0, 4, 2, 3, 1],
    "lying":   [0, 4, 3, 1, 2],
}
ARC_STEP_DEG      = 2.0
ARC_STEP_DELAY    = 0.15

# Tip correction, ruler-measured 2026-07-04: constant x/y offset, z droop
# proportional to horizontal reach. move_to() aims at target minus these.
TIP_ERROR_X_M   = -0.025
TIP_ERROR_Y_M   = 0.020
SAG_PER_M_REACH = 0.25

# Floor is 11.3cm below the deck (z=0); lower z targets are clamped.
DECK_ABOVE_FLOOR_M = 0.113

# Registry so the test tooling can jog to a full pose (arm + gripper) by name.
NAMED_POSES = {
    "home":   (HOME_ANGLES, GRIPPER_OPEN),
    "bin":    (BIN_DROP_ANGLES, GRIPPER_OPEN),
    "sweet1": (GRAB_ANGLES, GRIPPER_CLOSED),
}

class GraspPlanner:
    """
    Executes specific arm trajectories, converting 3D target coordinates
    into safe hardware actions using IK.
    
    Rule 1 & Rule 2 compliant: It reads IK outputs and routes them to 
    the actuator interface without talking to I2C/SPI directly.
    """
    def __init__(self):
        self.ik = AnalyticalArmIK()           # primary: closed-form solver
        self.kinematics = ArmKinematics()     # backup: numerical ikpy solver
        self.actuator = ArmActuator()         # Grabs hardware connection
        # Tracked pose seeds every stepped ramp. Prefer the pose persisted by
        # the PREVIOUS session (any tool/planner) over blindly assuming home —
        # a wrong start turns "stepped" moves into full-speed snaps.
        last = load_last_pose()
        if last is not None:
            self._arm, self._gripper = last[:5], last[5]
            print(f"[GraspPlanner] resuming last commanded pose: "
                  f"{[round(v, 1) for v in self._arm]} grip={self._gripper:.1f}")
        else:
            self._arm = list(HOME_ANGLES)     # no saved pose: assume home
            self._gripper = GRIPPER_OPEN

    def _move_stepped(self, target_arm, target_gripper):
        """Stepped move to a full pose, ending with a write-through of every
        channel (no joint feedback — tracking can be wrong)."""
        start = list(self._arm) + [self._gripper]
        target = list(target_arm) + [target_gripper]
        stepped_move(self.actuator, start, target)
        for ch, v in enumerate(target):
            self.actuator.set_channel_angle(ch, v)
        self._arm, self._gripper = list(target_arm), float(target_gripper)
        save_last_pose(self._arm + [self._gripper])

    def _move_one(self, ch: int, value: float):
        """Move one channel gently (ch 0-4 = CH1-5, ch 5 = gripper), ending
        with a write-through so the command is never silently dropped."""
        start = list(self._arm) + [self._gripper]
        target = list(start)
        target[ch] = clamp_channel_angle(ch, float(value))
        stepped_move(self.actuator, start, target, ARC_STEP_DEG, ARC_STEP_DELAY)
        self.actuator.set_channel_angle(ch, target[ch])
        if ch < 5:
            self._arm[ch] = target[ch]
        else:
            self._gripper = target[ch]
        save_last_pose(self._arm + [self._gripper])

    def grab_arc_pose(self, solved: list, tin_pose: str = "upright") -> bool:
        """Execute the tuned arc-grasp sequence at a solved [CH1..CH5] pose
        (from ArcGraspSolver): open, lift high, approach in the safe channel
        order FOR THIS TIN POSE (upright: shoulder last; lying: elbow last),
        close, lift shoulder — ends HOLDING the can. Follow with
        dump_to_bin() to deposit it. tin_pose: "upright" | "lying" | "axial"
        ("axial" grabs like lying)."""
        if solved is None or len(solved) < 5:
            print("[GraspPlanner] grab_arc_pose: invalid pose, arm NOT moved.")
            return False
        order = ARC_GRAB_ORDER["lying" if tin_pose in ("lying", "axial")
                               else "upright"]
        print(f"[GraspPlanner] Arc grab ({tin_pose}) at CH1-5 = {solved}")
        self.control_gripper("open")
        # 1. lift to a high/folded pose at the current azimuth so the base
        #    rotation and descent below can't drag the gripper on the floor.
        for ch in (1, 2, 3):                     # CH2 shoulder, CH3 elbow, CH4 wrist
            self._move_one(ch, ARC_LIFT_ARM[ch])
        # 2. approach in the pose-specific order; the last channel is the
        #    one that lowers onto the tin. CH5 is set while still high so a
        #    stale roll from a previous grab can't hit the floor.
        for ch in order:
            self._move_one(ch, float(solved[ch]))
        self.control_gripper("close")
        self._move_one(1, ARC_LIFT_ARM[1])       # lift shoulder back up, holding
        self.kinematics.reset_warm_start()
        return True

    def force_home(self) -> bool:
        """Drive to HOME gently; the ending write-through asserts the pose
        even when tracking was stale."""
        print("[GraspPlanner] Force-homing (stepped + write-through)...")
        self._move_stepped(HOME_ANGLES, GRIPPER_OPEN)
        self.kinematics.reset_warm_start()
        return True

    def move_to(self, target_xyz: list, tool_direction=GRIPPER_DOWN, solver="analytic",
                compensate: bool = True):
        """
        Calculates and moves the arm to the (x, y, z) position in meters.
        By default the gripper is kept pointing DOWN (tool_direction=GRIPPER_DOWN);
        pass tool_direction=None for pure position IK.
        solver="analytic" uses the closed-form IK (default); solver="ikpy" uses the
        numerical backup.
        compensate=True aims at target-minus-measured-error (constant x/y shift +
        reach-proportional z lift) so the REAL tip lands on target_xyz despite
        gravity sag; pass False to command the raw model target.
        """
        goal = list(target_xyz)
        if goal[2] < -DECK_ABOVE_FLOOR_M:
            print(f"[GraspPlanner] target z={goal[2]:.3f} is BELOW THE FLOOR "
                  f"(floor = -{DECK_ABOVE_FLOOR_M:.3f} from the deck); clamping to floor level.")
            goal[2] = -DECK_ABOVE_FLOOR_M
        if compensate:
            gx = goal[0] - TIP_ERROR_X_M
            gy = goal[1] - TIP_ERROR_Y_M
            gz = goal[2] + SAG_PER_M_REACH * math.hypot(gx, gy)
            goal = [gx, gy, gz]
            print(f"\n[GraspPlanner] Planning arm movement to {target_xyz} "
                  f"(sag-compensated aim {[round(v, 4) for v in goal]}, {solver}) ...")
        else:
            print(f"\n[GraspPlanner] Planning arm movement to {target_xyz} ({solver}) ...")

        if solver == "ikpy":
            servo_angles = self.kinematics.calculate_servo_angles(goal, tool_direction)
        else:
            if tool_direction is GRIPPER_DOWN:
                approach = "down"
            elif tool_direction is GRIPPER_UP:
                approach = "up"
            elif tool_direction is GRIPPER_LEVEL:
                approach = "level"
            else:
                approach = "free"   # None or arbitrary vectors (ikpy handles those)
            servo_angles = self.ik.solve(goal, approach=approach)

        # None means no reachable solution — do NOT move the arm and report honestly.
        if servo_angles is None:
            print(f"[GraspPlanner] no reachable IK solution for {target_xyz}; arm NOT moved.")
            return False

        print("[GraspPlanner] Kinematics Solved! Target Degrees (CH1-5):", servo_angles)

        # Send physical angles to the actuator as a gentle stepped move.
        self._move_stepped(servo_angles, self._gripper)
        return True

    def goto_named_pose(self, name: str):
        """Jog to one of the hardcoded NAMED_POSES (arm + gripper) by name —
        no IK, just servo commands. Used by tests/test_kinematics.py."""
        pose = NAMED_POSES.get(name)
        if pose is None:
            print(f"[GraspPlanner] Unknown pose '{name}'. Known: {list(NAMED_POSES)}")
            return False
        arm_angles, gripper = pose
        print(f"[GraspPlanner] Moving to '{name}': arm={arm_angles}, gripper={gripper}")
        self._move_stepped(arm_angles, gripper)
        self.kinematics.reset_warm_start()
        return True

    def home(self):
        """
        Return to the hardcoded home/stow pose (HOME_ANGLES) and open the gripper.
        Commands the servos directly (no IK).
        """
        print("[GraspPlanner] Homing to stow pose (servo-level, no IK)...")
        self._move_stepped(HOME_ANGLES, GRIPPER_OPEN)
        self.kinematics.reset_warm_start()
        return True

    def grab(self):
        print("[GraspPlanner] Grabbing at hardcoded grasp pose (no IK)...")
        self._move_stepped(GRAB_ANGLES, GRIPPER_OPEN)
        self.kinematics.reset_warm_start()
        return True

    def dump_to_bin(self):
        """
        Move to the fixed bin-drop pose and release the can. The bin is mounted on
        the chassis at a fixed spot, so this is a hardcoded servo pose (BIN_DROP_ANGLES),
        not an IK target. Capture/update those angles with tests/servo_jog.py.
        """
        print("[GraspPlanner] Moving to bin-drop pose (hardcoded, no IK)...")
        self._move_stepped(BIN_DROP_ANGLES, self._gripper)   # carry there, still holding
        self.kinematics.reset_warm_start()
        self.control_gripper("open")  # release the can into the bin
        return True

    def get_pose(self) -> dict:
        """Tracked pose for UIs: {'arm': [CH1..CH5], 'gripper': CH6}. This is
        the COMMANDED pose (servos have no feedback), valid as long as every
        move went through this planner."""
        return {"arm": list(self._arm), "gripper": self._gripper}

    def jog_channel(self, ch: int, delta_deg: float) -> float:
        """Nudge one channel by delta degrees (ch 0-4 = CH1-5, ch 5 = gripper),
        gently, clamped to 0-180. Returns the new commanded angle. Used by the
        web dashboard's manual arm control."""
        if not 0 <= ch <= 5:
            raise ValueError(f"channel {ch} out of range 0-5")
        current = self._gripper if ch == 5 else self._arm[ch]
        target = max(0.0, min(180.0, current + float(delta_deg)))
        self._move_one(ch, target)
        return target

    def control_gripper(self, action: str):
        """Separated gripper logic. 'close' closes on the can; 'open'/'neutral'/'stow'
        all open the jaws (same position on this gripper)."""
        if action == "close":
            print("[GraspPlanner] Closing Gripper...")
            self._gripper = GRIPPER_CLOSED
        elif action in ("open", "neutral", "stow"):
            print("[GraspPlanner] Opening Gripper...")
            self._gripper = GRIPPER_OPEN
        else:
            return
        self.actuator.set_gripper_angle(self._gripper)


# If the module is run independently, execute a test script 
if __name__ == "__main__":
    planner = GraspPlanner()
    # Test a forward coordinate: 20cm ahead, 10cm high.
    planner.move_to([0.20, 0.0, 0.10])
