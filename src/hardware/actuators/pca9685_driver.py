# src/hardware/actuators/pca9685_driver.py

# Pulse-width range (microseconds) for the YF-6125MG servos.
# adafruit_servokit defaults to 750–2250 µs; 500–2500 µs gives these servos their full
# physical sweep. Measured (tests/test_servo_travel.py): across 500–2500 µs they travel
# only ~180° — NOT the labelled 270°. So actuation_range is set to 180, which makes one
# commanded degree equal one physical degree (1:1). SERVO_RANGE_DEG must match
# SERVO_CMD_MAX in src/arm/kinematics.py.
SERVO_MIN_PULSE_US = 500
SERVO_MAX_PULSE_US = 2500
SERVO_RANGE_DEG = 180

try:
    from adafruit_servokit import ServoKit
except ImportError:
    # Fallback for testing environments without real hardware
    print("WARNING: adafruit_servokit not found. Running in simulation mode.")
    class DummyServo:
        def __init__(self):
            self.angle = 90
            self.actuation_range = SERVO_RANGE_DEG

        def set_pulse_width_range(self, min_pulse, max_pulse):
            self._pulse = (min_pulse, max_pulse)

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
        # Configure every channel: actuation_range = real travel (180°) AND the correct
        # pulse-width window, so a commanded degree maps 1:1 to that physical degree.
        for i in range(16):
            self.kit.servo[i].actuation_range = SERVO_RANGE_DEG
            self.kit.servo[i].set_pulse_width_range(SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US)

    def set_arm_angles(self, angles_deg: list):
        """
        Takes a list of 5 mapped angles (CH1-CH5) and writes them to the hardware.
        angles_deg: List of 5 floats representing absolute target angles (0-180).
        """
        for i, angle in enumerate(angles_deg):
            if i >= 5:
                break

            # Additional hardware safety bound block
            angle = max(0.0, min(SERVO_RANGE_DEG, angle))

            self.kit.servo[i].angle = angle
            print(f"[Actuator] CH{i+1} hardware written angle: {angle:.1f}°")

    def set_gripper_angle(self, angle_deg: float):
        """
        Controls the gripper separately (CH6). With actuation_range=180, the working
        open/close commands are ~40 (closed) .. ~120 (open); see GraspPlanner.
        """
        angle = max(0.0, min(SERVO_RANGE_DEG, angle_deg))
        self.kit.servo[5].angle = angle
        print(f"[Actuator] CH6 (Gripper) hardware written angle: {angle:.1f}°")

