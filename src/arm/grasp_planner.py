from src.arm.kinematics import ArmKinematics, GRIPPER_DOWN
from src.hardware.actuators.pca9685_driver import ArmActuator

# Hardcoded servo pose [CH1..CH5] that drops a can into the bin mounted on the chassis.
# The bin is fixed relative to the arm base, so this pose never changes -> no IK needed.
# CAPTURE IT: jog the arm over the bin with tests/servo_jog.py, press 'p', and paste the
# CH1-5 numbers here. (Still neutral = placeholder; update before relying on dump_to_bin.)
BIN_DROP_ANGLES = [135.0, 135.0, 135.0, 135.0, 135.0]

class GraspPlanner:
    """
    Executes specific arm trajectories, converting 3D target coordinates
    into safe hardware actions using IK.
    
    Rule 1 & Rule 2 compliant: It reads IK outputs and routes them to 
    the actuator interface without talking to I2C/SPI directly.
    """
    def __init__(self):
        self.kinematics = ArmKinematics()
        self.actuator = ArmActuator()  # Grabs hardware connection

    def move_to(self, target_xyz: list, tool_direction=GRIPPER_DOWN):
        """
        Calculates and moves the arm to the (x, y, z) position in meters.
        By default the gripper is kept pointing DOWN (tool_direction=GRIPPER_DOWN);
        pass tool_direction=None for pure position IK.
        """
        print(f"\n[GraspPlanner] Planning arm movement to {target_xyz} ...")

        # Compute angles using ikpy (This already applies proper Offsets and Physical Bounds via kinematics.py)
        servo_angles = self.kinematics.calculate_servo_angles(target_xyz, tool_direction)

        # None means IK did not converge — do NOT move the arm and report honestly.
        if servo_angles is None:
            print(f"[GraspPlanner] ⚠️ No reachable IK solution for {target_xyz}; arm NOT moved.")
            return False

        print("[GraspPlanner] Kinematics Solved! Target Degrees (CH1-5):", servo_angles)

        # Send physical 0~270 angles to actuator
        self.actuator.set_arm_angles(servo_angles)
        return True

    def home(self):
        """
        Return to the neutral stow pose (all joints centered, arm straight up).
        Commands the servos directly — the straight-up pose cannot satisfy the
        gripper-down constraint, so routing it through IK would (correctly) fail.
        """
        print("[GraspPlanner] Homing to neutral pose (servo-level, no IK)...")
        self.actuator.set_arm_angles([135.0, 135.0, 135.0, 135.0, 135.0])
        self.kinematics.reset_warm_start()
        return True

    def dump_to_bin(self):
        """
        Move to the fixed bin-drop pose and release the can. The bin is mounted on
        the chassis at a fixed spot, so this is a hardcoded servo pose (BIN_DROP_ANGLES),
        not an IK target. Capture/update those angles with tests/servo_jog.py.
        """
        print("[GraspPlanner] Moving to bin-drop pose (hardcoded, no IK)...")
        self.actuator.set_arm_angles(BIN_DROP_ANGLES)
        self.kinematics.reset_warm_start()
        self.control_gripper("open")  # release the can into the bin
        return True

    def control_gripper(self, action: str):
        """
        Separated gripper logic: open, close, stow.
        Physical bounds: 60 to 180 degrees.
        """
        if action == "open":
            print("[GraspPlanner] Opening Gripper...")
            self.actuator.set_gripper_angle(180.0)
        elif action == "close":
            print("[GraspPlanner] Closing Gripper...")
            self.actuator.set_gripper_angle(60.0)
        elif action == "neutral" or action == "stow":
            print("[GraspPlanner] Gripper to Neutral...")
            self.actuator.set_gripper_angle(120.0)


# If the module is run independently, execute a test script 
if __name__ == "__main__":
    planner = GraspPlanner()
    # Test a forward coordinate: 20cm ahead, 10cm high.
    planner.move_to([0.20, 0.0, 0.10])
