from typing import Protocol

from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import WheelCommand

# ZK-BM1 accepts PWM up to ~2 kHz (board limit, guarded below). But RPi.GPIO
# PWM is SOFTWARE-timed: above a few hundred Hz the duty cycle jitters badly
# (at 1 kHz the period is 1 ms; ±0.1 ms scheduler jitter = ±10% duty error,
# worse while YOLO loads the CPU). 200 Hz keeps the duty accurate.
PWM_FREQ_HZ = 200
ZKBM1_MAX_PWM_HZ = 2000

# Static friction floor: duty below this buzzes without turning. 0 = off.
MIN_MOVE_DUTY = 0.0   # TODO tune on hardware: raise until the chassis creeps


def _apply_stiction_floor(left: float, right: float):
    """Scale a too-weak wheel pair up together so the faster wheel clears
    MIN_MOVE_DUTY, keeping left/right ratio (and so the steering) intact.

    The slower wheel may still land under the floor -- that is wanted: a
    dragging inner wheel tightens the arc, which is what a low-speed turn
    physically is. A stopped pair stays stopped.
    """
    peak = max(abs(left), abs(right))
    if MIN_MOVE_DUTY <= 0.0 or peak <= 0.0 or peak >= MIN_MOVE_DUTY:
        return left, right
    gain = MIN_MOVE_DUTY / peak
    return left * gain, right * gain


class PWMProtocol(Protocol):
    def start(self, value: float) -> None: ...

    def ChangeDutyCycle(self, value: float) -> None: ...

    def stop(self) -> None: ...


class MockPWM:
    def __init__(self, _pin: int, _freq: int) -> None:
        self.value = 0.0

    def start(self, value: float) -> None:
        self.value = float(value)

    def ChangeDutyCycle(self, value: float) -> None:
        self.value = float(value)

    def stop(self) -> None:
        self.value = 0.0


class MockGPIO:
    BCM = 1
    OUT = 1
    LOW = 0

    def setmode(self, _mode: int) -> None:
        return

    def setup(self, _pins, _mode: int) -> None:
        return

    def output(self, _pins, _value) -> None:
        return

    def PWM(self, pin: int, freq: int) -> PWMProtocol:
        return MockPWM(pin, freq)

    def cleanup(self, _pins=None) -> None:
        return


try:
    import RPi.GPIO as GPIO  # type: ignore
except Exception:  # pragma: no cover
    GPIO = MockGPIO()  # type: ignore


class _PigpioPWM:
    """One pin's PWM through the pigpiod daemon. DMA-timed in the daemon
    process, so the duty stays accurate while YOLO loads the CPU — the
    soft-PWM jitter RPi.GPIO suffers does not apply. Same interface as
    RPi.GPIO's PWM object so the driver logic below is backend-agnostic."""

    # pigpio duty range per cycle; at 200 Hz the daemon resolves 1000 steps.
    _RANGE = 1000

    def __init__(self, pi, pin: int, freq: int) -> None:
        self._pi = pi
        self._pin = pin
        pi.set_mode(pin, 1)                  # pigpio.OUTPUT
        pi.set_PWM_frequency(pin, freq)
        pi.set_PWM_range(pin, self._RANGE)
        pi.set_PWM_dutycycle(pin, 0)

    def start(self, value: float) -> None:
        self.ChangeDutyCycle(value)

    def ChangeDutyCycle(self, value: float) -> None:
        duty = int(round(max(0.0, min(100.0, float(value))) * self._RANGE / 100.0))
        self._pi.set_PWM_dutycycle(self._pin, duty)

    def stop(self) -> None:
        self._pi.set_PWM_dutycycle(self._pin, 0)


def _connect_pigpio():
    """A pigpiod connection, or None (daemon not running / not installed /
    Pi 5, where pigpio is unsupported). None -> RPi.GPIO soft PWM fallback."""
    try:
        import pigpio  # type: ignore
        pi = pigpio.pi()
        if pi.connected:
            return pi
    except Exception:
        pass
    return None


class PWMActuator:
    """Hardware module: converts wheel command into GPIO + PWM signals.
    Implements src.hardware.actuators.interfaces.ActuatorInterface.

    Wired for the ZK-BM1 dual H-bridge, which has NO ENA/ENB enable pins.
    Unlike an L298N (separate direction inputs + a PWM enable line), the
    ZK-BM1 sets both direction AND speed on the two input pins per motor:
    to run a motor, PWM one input and hold the other LOW; the duty cycle IS
    the speed. So this driver keeps a PWM channel on all four input pins.
        Left  motor (A): in1 / in2
        Right motor (B): in3 / in4
    """

    def __init__(self, pins: MotorPins | None = None, calibration: MotionCalibration | None = None, pwm_freq: int = PWM_FREQ_HZ) -> None:
        if pwm_freq > ZKBM1_MAX_PWM_HZ:
            raise ValueError(f"pwm_freq {pwm_freq} Hz exceeds the ZK-BM1's "
                             f"~{ZKBM1_MAX_PWM_HZ} Hz input limit")
        self.pins = pins or MotorPins()
        self.cal = calibration or MotionCalibration()
        self.last_duty = (0.0, 0.0)
        self._my_pins = [self.pins.in1, self.pins.in2, self.pins.in3, self.pins.in4]

        # Prefer pigpiod (DMA-timed, duty immune to CPU load); fall back to
        # RPi.GPIO soft PWM when the daemon is not available.
        self._pi = _connect_pigpio()
        if self._pi is not None:
            self.pwm_in1 = _PigpioPWM(self._pi, self.pins.in1, pwm_freq)
            self.pwm_in2 = _PigpioPWM(self._pi, self.pins.in2, pwm_freq)
            self.pwm_in3 = _PigpioPWM(self._pi, self.pins.in3, pwm_freq)
            self.pwm_in4 = _PigpioPWM(self._pi, self.pins.in4, pwm_freq)
            print("[PWMActuator] pigpio backend (hardware-timed PWM)")
            return

        # Release ONLY OUR OWN pins from a previous unclean run. A global
        # GPIO.cleanup() here would tear down every other module's setup in
        # this process — the ultrasonic's TRIG/ECHO pins are configured
        # BEFORE the motor driver in the runtime, so a global cleanup made
        # every later distance read fail.
        try:
            GPIO.cleanup(self._my_pins)  # type: ignore
        except Exception:
            pass

        try:
            GPIO.setmode(GPIO.BCM)  # type: ignore
        except RuntimeError:
            # GPIO mode already set, that's fine
            pass

        GPIO.setup(self._my_pins, GPIO.OUT)

        # One PWM channel per input pin (ZK-BM1 has no separate enable line).
        self.pwm_in1 = GPIO.PWM(self.pins.in1, pwm_freq)
        self.pwm_in2 = GPIO.PWM(self.pins.in2, pwm_freq)
        self.pwm_in3 = GPIO.PWM(self.pins.in3, pwm_freq)
        self.pwm_in4 = GPIO.PWM(self.pins.in4, pwm_freq)
        for pwm in (self.pwm_in1, self.pwm_in2, self.pwm_in3, self.pwm_in4):
            pwm.start(0)

    def apply(self, command: WheelCommand) -> None:
        left_speed = command.left_speed
        right_speed = command.right_speed

        # Motors plugged into each other's channels -> send each speed to the
        # other side. This runs FIRST because trim and invert describe the
        # CHANNEL, not the command: applying them before the swap put the
        # straightness trim on the wrong wheel, and left a pure pivot (equal
        # and opposite speeds) with two identical values, so swapping them
        # afterwards did nothing at all and turns stayed mirrored.
        if self.cal.swap_left_right:
            left_speed, right_speed = right_speed, left_speed

        if command.apply_trim:
            if command.trim_set == "forward":
                if left_speed > 0:
                    left_speed *= self.cal.motor_a_forward_trim
                if right_speed > 0:
                    right_speed *= self.cal.motor_b_forward_trim
            elif command.trim_set == "backward":
                if left_speed < 0:
                    left_speed *= self.cal.motor_a_backward_trim
                if right_speed < 0:
                    right_speed *= self.cal.motor_b_backward_trim
            elif command.trim_set == "turn":
                left_speed *= self.cal.motor_a_turn_trim
                right_speed *= self.cal.motor_b_turn_trim

        left_speed, right_speed = _apply_stiction_floor(left_speed, right_speed)

        left_speed = max(-100.0, min(100.0, left_speed))
        right_speed = max(-100.0, min(100.0, right_speed))

        # Fix reversed motor wiring: flip polarity so the wheel turns the way the
        # command (and the on-screen suggestion) means it to.
        if self.cal.invert_left:
            left_speed = -left_speed
        if self.cal.invert_right:
            right_speed = -right_speed

        # Telemetry for bench tools: the duty actually sent, after swap, trim,
        # stiction floor, clamp and invert.
        self.last_duty = (left_speed, right_speed)
        self._set_left(left_speed)
        self._set_right(right_speed)

    def stop(self) -> None:
        """COAST ('Free Running Motor Stop'): both inputs of each motor LOW,
        motor windings open. The wheels are free to spin — an external push
        (e.g. the arm shaking the chassis) can roll the robot out of position.
        Use brake() to hold."""
        self.last_duty = (0.0, 0.0)
        for pwm in (self.pwm_in1, self.pwm_in2, self.pwm_in3, self.pwm_in4):
            pwm.ChangeDutyCycle(0)

    def brake(self) -> None:
        """ACTIVE BRAKE ('Fast Motor Stop'): both inputs of each motor driven
        HIGH (PWM 100%) at the same time. This shorts the motor windings, so
        any attempt to turn the wheel — a push forward or back — induces a
        current that opposes the motion (dynamic braking). The robot resists
        being rolled, holding position while the arm actuates. No mechanical
        brake exists; this is the strongest hold the hardware allows.
        Stationary, it draws ~no current; current only flows while something
        is actively trying to move it."""
        for pwm in (self.pwm_in1, self.pwm_in2, self.pwm_in3, self.pwm_in4):
            pwm.ChangeDutyCycle(100)

    def close(self) -> None:
        self.stop()
        for pwm in (self.pwm_in1, self.pwm_in2, self.pwm_in3, self.pwm_in4):
            pwm.stop()
        if self._pi is not None:
            try:
                self._pi.stop()          # disconnect from pigpiod only
            except Exception:
                pass
            return
        # Own pins only — a global cleanup wipes the ultrasonic's setup too.
        try:
            GPIO.cleanup(self._my_pins)  # type: ignore
        except Exception:
            pass

    def _set_left(self, speed: float) -> None:
        # ZK-BM1: PWM the forward input for +speed, the reverse input for
        # -speed; the idle input is held at 0% (LOW). Duty cycle = speed.
        # ORDER MATTERS: drop the idle input to 0 BEFORE raising the active
        # one — otherwise a direction change passes through a moment with
        # BOTH inputs high, which the board treats as a brake pulse.
        if speed > 0:
            self.pwm_in2.ChangeDutyCycle(0)
            self.pwm_in1.ChangeDutyCycle(abs(speed))
        elif speed < 0:
            self.pwm_in1.ChangeDutyCycle(0)
            self.pwm_in2.ChangeDutyCycle(abs(speed))
        else:
            self.pwm_in1.ChangeDutyCycle(0)
            self.pwm_in2.ChangeDutyCycle(0)

    def _set_right(self, speed: float) -> None:
        if speed > 0:
            self.pwm_in4.ChangeDutyCycle(0)
            self.pwm_in3.ChangeDutyCycle(abs(speed))
        elif speed < 0:
            self.pwm_in3.ChangeDutyCycle(0)
            self.pwm_in4.ChangeDutyCycle(abs(speed))
        else:
            self.pwm_in3.ChangeDutyCycle(0)
            self.pwm_in4.ChangeDutyCycle(0)
