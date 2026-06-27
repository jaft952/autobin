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


def stepped_move(actuator, start, target, step_deg=5.0, step_delay=0.15, instant=()):
    """Drive servos from `start` to `target` angle lists (index 0 = CH1) gently.

    Every NON-`instant` channel steps TOGETHER so none moves more than `step_deg`
    per step, pausing `step_delay` between steps (low current draw, no slamming).
    Channels listed in `instant` snap straight to their target — e.g. the gripper
    (CH6) — and channels whose angle doesn't change are skipped. Returns a copy of
    `target` so the caller can track the new pose.

    Shared by the grasp (test_ibvs_centering), the servo jog tool, and GraspPlanner
    so they all move the arm at one consistent, gentle speed. Writes are clamped to
    [0, SERVO_RANGE_DEG].
    """
    import time
    n = min(len(start), len(target))

    def write(ch, val):
        actuator.kit.servo[ch].angle = max(0.0, min(SERVO_RANGE_DEG, val))

    for ch in instant:
        if ch < n:
            write(ch, target[ch])
    moving = [ch for ch in range(n)
              if ch not in instant and abs(target[ch] - start[ch]) > 1e-9]
    if moving:
        span = max(abs(target[ch] - start[ch]) for ch in moving)
        steps = max(1, int((span + step_deg - 1e-6) // step_deg))
        for k in range(1, steps + 1):
            f = k / steps
            for ch in moving:
                write(ch, start[ch] + (target[ch] - start[ch]) * f)
            time.sleep(step_delay)
    return list(target)

