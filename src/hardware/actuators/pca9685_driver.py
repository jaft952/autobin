SERVO_MIN_PULSE_US = 500
SERVO_MAX_PULSE_US = 2500
SERVO_RANGE_DEG = 180

GRIPPER_MIN_DEG = 0.0
GRIPPER_MAX_DEG = 78.0
CHANNEL_ANGLE_LIMITS = {5: (GRIPPER_MIN_DEG, GRIPPER_MAX_DEG)}


def clamp_channel_angle(ch: int, angle_deg: float) -> float:
    """Keep a channel angle inside its safe range."""
    angle = max(0.0, min(float(SERVO_RANGE_DEG), float(angle_deg)))
    lo, hi = CHANNEL_ANGLE_LIMITS.get(ch, (0.0, float(SERVO_RANGE_DEG)))
    return max(lo, min(hi, angle))


def _last_pose_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "arm" / "config" / "last_pose.json"


def save_last_pose(angles6):
    """Save the last arm pose to a file."""
    try:
        import json
        p = _last_pose_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([round(float(a), 1) for a in angles6[:6]]))
    except Exception:
        pass


def load_last_pose():
    """Load the last arm pose, or None."""
    try:
        import json
        vals = json.loads(_last_pose_path().read_text())
        if isinstance(vals, list) and len(vals) == 6:
            return [max(0.0, min(float(SERVO_RANGE_DEG), float(v))) for v in vals]
    except Exception:
        pass
    return None

try:
    from adafruit_servokit import ServoKit  # type: ignore
except ImportError:
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
        """Write five angles to CH1 to CH5."""
        for i, angle in enumerate(angles_deg):
            if i >= 5:
                break

            angle = clamp_channel_angle(i, angle)

            self.kit.servo[i].angle = angle  # type: ignore
            print(f"[Actuator] CH{i+1} hardware written angle: {angle:.1f}°")

    def set_channel_angle(self, ch: int, angle_deg: float):
        """Write one channel angle directly."""
        self.kit.servo[ch].angle = clamp_channel_angle(ch, angle_deg)  # type: ignore

    def set_gripper_angle(self, angle_deg: float):
        """Set the gripper angle within its safe range."""
        angle = clamp_channel_angle(5, angle_deg)
        self.kit.servo[5].angle = angle  # type: ignore
        print(f"[Actuator] CH6 (Gripper) hardware written angle: {angle:.1f}°")

    def release(self):
        """Turn off all servos so the arm goes limp."""
        for i in range(6):
            self.kit.servo[i].angle = None


SERVO_UPDATE_HZ = 50.0


def stepped_move(actuator, start, target, step_deg=5.0, step_delay=0.15, instant=()):
    """Move servos smoothly over time."""
    import math
    import time
    n = min(len(start), len(target))

    def write(ch, val):
        actuator.kit.servo[ch].angle = clamp_channel_angle(ch, val)

    for ch in instant:
        if ch < n:
            write(ch, target[ch])
    moving = [ch for ch in range(n)
              if ch not in instant and abs(target[ch] - start[ch]) > 1e-9]
    if not moving:
        return list(target)

    span = max(abs(target[ch] - start[ch]) for ch in moving)
    speed_dps = step_deg / max(step_delay, 1e-6)
    duration = max(span / max(speed_dps, 1e-6), 2.0 / SERVO_UPDATE_HZ)
    ticks = max(2, int(round(duration * SERVO_UPDATE_HZ)))
    period = duration / ticks

    t0 = time.perf_counter()
    for k in range(1, ticks + 1):
        f = k / ticks
        eased = 0.5 - 0.5 * math.cos(math.pi * f)
        for ch in moving:
            write(ch, start[ch] + (target[ch] - start[ch]) * eased)
        rest = t0 + k * period - time.perf_counter()
        if rest > 0:
            time.sleep(rest)
    return list(target)

