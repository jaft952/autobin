from src.arm.kinematics import ArmKinematics, GRIPPER_DOWN
from src.hardware.actuators.pca9685_driver import ArmActuator

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
