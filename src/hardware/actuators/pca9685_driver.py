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

    def set_channel_angle(self, ch: int, angle_deg: float):
        """Write ONE channel directly (0-4 = CH1-5 arm, 5 = CH6 gripper),
        quietly. Used as the WRITE-THROUGH guarantee after stepped moves:
        a commanded channel must always receive its target angle, even when
        the software's tracked pose claims it's already there (there is no
        joint feedback — tracking can be wrong, e.g. right after boot)."""
        angle = max(0.0, min(SERVO_RANGE_DEG, angle_deg))
        self.kit.servo[ch].angle = angle

    def set_gripper_angle(self, angle_deg: float):
        """
        Controls the gripper separately (CH6). With actuation_range=180, the working
        open/close commands are ~40 (closed) .. ~120 (open); see GraspPlanner.
        """
        angle = max(0.0, min(SERVO_RANGE_DEG, angle_deg))
        self.kit.servo[5].angle = angle
        print(f"[Actuator] CH6 (Gripper) hardware written angle: {angle:.1f}°")


# Hobby servos refresh their PWM at ~50 Hz — stream setpoints at that rate so
# the servo follows the trajectory CONTINUOUSLY instead of sprinting to each
# coarse step and slamming to a halt.
SERVO_UPDATE_HZ = 50.0


def stepped_move(actuator, start, target, step_deg=5.0, step_delay=0.15, instant=()):
    """Smooth time-based move (v2, 2026-07-08).

    The old implementation jumped step_deg degrees then slept step_delay —
    the servo sprinted each jump at full internal speed and slammed to a
    stop, N times per move. Those repeated jolts excited the arm's
    resonance: visible shaking, loosening screws, chassis kicks. (A LONGER
    delay made it worse, not gentler — the jolts just became more distinct.)

    Now the same (step_deg, step_delay) pair is interpreted as a SPEED
    (step_deg degrees per step_delay seconds — callers keep their tuned
    pace and total duration), but execution streams fine-grained setpoints
    at SERVO_UPDATE_HZ with cosine ease-in/ease-out: the servo never gets
    ahead of the target, so it moves continuously, and acceleration ramps
    gently at both ends instead of jerking.
    """
    import math
    import time
    n = min(len(start), len(target))

    def write(ch, val):
        actuator.kit.servo[ch].angle = max(0.0, min(SERVO_RANGE_DEG, val))

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

