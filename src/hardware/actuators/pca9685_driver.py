SERVO_MIN_PULSE_US = 500
SERVO_MAX_PULSE_US = 2500
SERVO_RANGE_DEG = 180

# Gripper (CH6) servo travel is narrower than the linkage; clamp keeps writes inside its safe window.
GRIPPER_MIN_DEG = 0.0
GRIPPER_MAX_DEG = 78.0
CHANNEL_ANGLE_LIMITS = {5: (GRIPPER_MIN_DEG, GRIPPER_MAX_DEG)}


def clamp_channel_angle(ch: int, angle_deg: float) -> float:
    """0-180 hardware clamp plus the per-channel safe window above."""
    angle = max(0.0, min(float(SERVO_RANGE_DEG), float(angle_deg)))
    lo, hi = CHANNEL_ANGLE_LIMITS.get(ch, (0.0, float(SERVO_RANGE_DEG)))
    return max(lo, min(hi, angle))

# Persist last commanded pose so a ramp starts from the real last position, not an assumed home.


def _last_pose_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "arm" / "config" / "last_pose.json"


def save_last_pose(angles6):
    """Persist [CH1..CH5, gripper] after a completed move. Best-effort."""
    try:
        import json
        p = _last_pose_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([round(float(a), 1) for a in angles6[:6]]))
    except Exception:
        pass                                  # persistence must never break a move


def load_last_pose():
    """[CH1..CH5, gripper] from the previous session, or None."""
    try:
        import json
        vals = json.loads(_last_pose_path().read_text())
        if isinstance(vals, list) and len(vals) == 6:
            return [max(0.0, min(float(SERVO_RANGE_DEG), float(v))) for v in vals]
    except Exception:
        pass
    return None

try:
    from adafruit_servokit import ServoKit # type: ignore
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
        """Write 5 mapped angles (CH1-CH5) to the hardware."""
        for i, angle in enumerate(angles_deg):
            if i >= 5:
                break

            angle = clamp_channel_angle(i, angle)

            self.kit.servo[i].angle = angle # type: ignore
            print(f"[Actuator] CH{i+1} hardware written angle: {angle:.1f}°")

    def set_channel_angle(self, ch: int, angle_deg: float):
        """Write one channel directly (0-4 = CH1-5, 5 = gripper); always writes, even if tracked pose claims it's already there."""
        self.kit.servo[ch].angle = clamp_channel_angle(ch, angle_deg) # type: ignore

    def set_gripper_angle(self, angle_deg: float):
        """Controls the gripper (CH6), clamped to its safe window."""
        angle = clamp_channel_angle(5, angle_deg)
        self.kit.servo[5].angle = angle # type: ignore
        print(f"[Actuator] CH6 (Gripper) hardware written angle: {angle:.1f}°")

    def release(self):
        """Cut PWM to CH1-6 so servos go limp."""
        for i in range(6):
            self.kit.servo[i].angle = None


# Stream setpoints at the servo refresh rate for continuous motion.
SERVO_UPDATE_HZ = 50.0


def stepped_move(actuator, start, target, step_deg=5.0, step_delay=0.15, instant=()):
    """Smooth time-based move: streams eased setpoints at SERVO_UPDATE_HZ instead of jumping in coarse steps."""
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
    speed_dps = step_deg / max(step_delay, 1e-6)       # old pace, kept exactly
    duration = max(span / max(speed_dps, 1e-6), 2.0 / SERVO_UPDATE_HZ)
    ticks = max(2, int(round(duration * SERVO_UPDATE_HZ)))
    period = duration / ticks

    t0 = time.perf_counter()
    for k in range(1, ticks + 1):
        f = k / ticks
        eased = 0.5 - 0.5 * math.cos(math.pi * f)      # ease-in / ease-out
        for ch in moving:
            write(ch, start[ch] + (target[ch] - start[ch]) * eased)
        rest = t0 + k * period - time.perf_counter()
        if rest > 0:
            time.sleep(rest)                            # pace to wall clock
    return list(target)
