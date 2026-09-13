from typing import Protocol

from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import WheelCommand

PWM_FREQ_HZ = 200
ZKBM1_MAX_PWM_HZ = 2000

MIN_MOVE_DUTY = 0.0


def _apply_stiction_floor(left: float, right: float):
    """Raise weak wheel speeds so the wheels actually start turning."""
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
    """PWM on one pin using the pigpio daemon."""

    _RANGE = 1000

    def __init__(self, pi, pin: int, freq: int) -> None:
        self._pi = pi
        self._pin = pin
        pi.set_mode(pin, 1)
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
    """Connect to pigpio, or return None if not available."""
    try:
        import pigpio  # type: ignore
        pi = pigpio.pi()
        if pi.connected:
            return pi
    except Exception:
        pass
    return None


class PWMActuator:
    """Wheel driver for the ZK-BM1 motor board."""

    def __init__(self, pins: MotorPins | None = None, calibration: MotionCalibration | None = None, pwm_freq: int = PWM_FREQ_HZ) -> None:
        if pwm_freq > ZKBM1_MAX_PWM_HZ:
            raise ValueError(f"pwm_freq {pwm_freq} Hz exceeds the ZK-BM1's "
                             f"~{ZKBM1_MAX_PWM_HZ} Hz input limit")
        self.pins = pins or MotorPins()
        self.cal = calibration or MotionCalibration()
        self.last_duty = (0.0, 0.0)
        self._my_pins = [self.pins.in1, self.pins.in2, self.pins.in3, self.pins.in4]
        self.last_pin_duty = {pin: 0.0 for pin in self._my_pins}

        self._pi = _connect_pigpio()
        if self._pi is not None:
            self.pwm_in1 = _PigpioPWM(self._pi, self.pins.in1, pwm_freq)
            self.pwm_in2 = _PigpioPWM(self._pi, self.pins.in2, pwm_freq)
            self.pwm_in3 = _PigpioPWM(self._pi, self.pins.in3, pwm_freq)
            self.pwm_in4 = _PigpioPWM(self._pi, self.pins.in4, pwm_freq)
            print("[PWMActuator] pigpio backend (hardware-timed PWM)")
            return

        try:
            GPIO.cleanup(self._my_pins)  # type: ignore
        except Exception:
            pass

        try:
            GPIO.setmode(GPIO.BCM)  # type: ignore
        except RuntimeError:
            pass

        GPIO.setup(self._my_pins, GPIO.OUT)

        self.pwm_in1 = GPIO.PWM(self.pins.in1, pwm_freq)
        self.pwm_in2 = GPIO.PWM(self.pins.in2, pwm_freq)
        self.pwm_in3 = GPIO.PWM(self.pins.in3, pwm_freq)
        self.pwm_in4 = GPIO.PWM(self.pins.in4, pwm_freq)
        for pwm in (self.pwm_in1, self.pwm_in2, self.pwm_in3, self.pwm_in4):
            pwm.start(0)

    def apply(self, command: WheelCommand) -> None:
        left_speed = command.left_speed
        right_speed = command.right_speed

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

        if self.cal.invert_left:
            left_speed = -left_speed
        if self.cal.invert_right:
            right_speed = -right_speed

        self.last_duty = (left_speed, right_speed)
        self._set_left(left_speed)
        self._set_right(right_speed)

    def _write(self, pwm, pin: int, value: float) -> None:
        pwm.ChangeDutyCycle(value)
        self.last_pin_duty[pin] = value

    def stop(self) -> None:
        """Coast: cut motor power and let the wheels roll."""
        self.last_duty = (0.0, 0.0)
        for pwm, pin in self._channels():
            self._write(pwm, pin, 0)

    def brake(self) -> None:
        """Brake: hold the motors to stop quickly."""
        self.last_duty = (0.0, 0.0)
        for pwm, pin in self._channels():
            self._write(pwm, pin, 0)
        for pwm, pin in self._channels():
            self._write(pwm, pin, 100)

    def close(self) -> None:
        self.stop()
        for pwm in (self.pwm_in1, self.pwm_in2, self.pwm_in3, self.pwm_in4):
            pwm.stop()
        if self._pi is not None:
            try:
                self._pi.stop()
            except Exception:
                pass
            return
        try:
            GPIO.cleanup(self._my_pins)  # type: ignore
        except Exception:
            pass

    def _channels(self):
        return ((self.pwm_in1, self.pins.in1), (self.pwm_in2, self.pins.in2),
                (self.pwm_in3, self.pins.in3), (self.pwm_in4, self.pins.in4))

    def _set_left(self, speed: float) -> None:
        if speed > 0:
            self._write(self.pwm_in2, self.pins.in2, 0)
            self._write(self.pwm_in1, self.pins.in1, abs(speed))
        elif speed < 0:
            self._write(self.pwm_in1, self.pins.in1, 0)
            self._write(self.pwm_in2, self.pins.in2, abs(speed))
        else:
            self._write(self.pwm_in1, self.pins.in1, 0)
            self._write(self.pwm_in2, self.pins.in2, 0)

    def _set_right(self, speed: float) -> None:
        if speed > 0:
            self._write(self.pwm_in4, self.pins.in4, 0)
            self._write(self.pwm_in3, self.pins.in3, abs(speed))
        elif speed < 0:
            self._write(self.pwm_in3, self.pins.in3, 0)
            self._write(self.pwm_in4, self.pins.in4, abs(speed))
        else:
            self._write(self.pwm_in3, self.pins.in3, 0)
            self._write(self.pwm_in4, self.pins.in4, 0)
