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
    def __init__(self):

        self.kit = ServoKit(channels=16)

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

