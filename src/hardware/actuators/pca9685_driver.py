# src/hardware/actuators/pca9685_driver.py
try:
    from adafruit_servokit import ServoKit
except ImportError:
    # Fallback for testing environments without real hardware
    print("WARNING: adafruit_servokit not found. Running in simulation mode.")
    class DummyServo:
        def __init__(self):
            self.angle = 135
            self.actuation_range = 270

    class ServoKit:
        def __init__(self, channels):
            self.servo = [DummyServo() for _ in range(channels)]


class ArmActuator:
    """
    Hardware Interface for the YF-6125MG servos connected via PCA9685.
    Strictly isolated from higher layer logic according to Subsumption Constraints (Rule 2).
    """
    def __init__(self):
        # Initialize the PC9685 I2C bus and 16-channel servo controller
        self.kit = ServoKit(channels=16)
        # Set all servo actuation ranges to 270 degrees per hardware spec
        for i in range(16):
            self.kit.servo[i].actuation_range = 270

    def set_arm_angles(self, angles_deg: list):
        """
        Takes a list of 5 mapped angles (CH1-CH5) and writes them to the hardware.
        angles_deg: List of 5 floats representing absolute target angles (0-270).
        """
        for i, angle in enumerate(angles_deg):
            if i >= 5:
                break
            
            # Additional hardware safety bound block
            angle = max(0.0, min(270.0, angle))
            
            self.kit.servo[i].angle = angle
            print(f"[Actuator] CH{i+1} hardware written angle: {angle:.1f}°")

    def set_gripper_angle(self, angle_deg: float):
        """
        Controls the gripper separately (CH6).
        angle_deg: target angle typically between 60.0 and 180.0
        """
        angle = max(60.0, min(180.0, angle_deg))
        self.kit.servo[5].angle = angle
        print(f"[Actuator] CH6 (Gripper) hardware written angle: {angle:.1f}°")

