# src/hardware/actuators/pca9685_driver.py

# Pulse-width range (microseconds) for the YF-6125MG 270° servos.
# IMPORTANT: adafruit_servokit defaults to 750–2250 µs, which is tuned for 180° servos.
# Using that default on a 270° servo compresses the travel — the servo only sweeps part
# of the commanded range, so commanded degrees != physical degrees (the arm barely bends).
# 500–2500 µs is the typical full-range spec for 270° hobby servos. Verify on the real
# arm with tests/test_servo_travel.py and adjust these two numbers if the sweep isn't
# exactly 270° between command 0 and command 270.
SERVO_MIN_PULSE_US = 500
SERVO_MAX_PULSE_US = 2500
SERVO_RANGE_DEG = 270

try:
    from adafruit_servokit import ServoKit
except ImportError:
    # Fallback for testing environments without real hardware
    print("WARNING: adafruit_servokit not found. Running in simulation mode.")
    class DummyServo:
        def __init__(self):
            self.angle = 135
            self.actuation_range = 270

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
        # Configure every channel for the 270° servos: full travel range AND correct
        # pulse-width window, so a commanded angle actually maps to that physical angle.
        for i in range(16):
            self.kit.servo[i].actuation_range = SERVO_RANGE_DEG
            self.kit.servo[i].set_pulse_width_range(SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US)

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

