from typing import Protocol

from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import WheelCommand


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

    def cleanup(self) -> None:
        return


try:
    import RPi.GPIO as GPIO  # type: ignore
except Exception:  # pragma: no cover
    GPIO = MockGPIO()  # type: ignore


class PWMActuator:
    """Hardware module: converts wheel command into GPIO + PWM signals."""

    def __init__(self, pins: MotorPins | None = None, calibration: MotionCalibration | None = None, pwm_freq: int = 100) -> None:
        self.pins = pins or MotorPins()
        self.cal = calibration or MotionCalibration()

        # Clean up GPIO state from previous runs, then set mode
        try:
            GPIO.cleanup()  # type: ignore
        except Exception:
            pass

        try:
            GPIO.setmode(GPIO.BCM)  # type: ignore
        except RuntimeError:
            # GPIO mode already set, that's fine
            pass

        GPIO.setup([self.pins.in1, self.pins.in2, self.pins.in3, self.pins.in4, self.pins.ena, self.pins.enb], GPIO.OUT)

        self.pwm_a = GPIO.PWM(self.pins.ena, pwm_freq)
        self.pwm_b = GPIO.PWM(self.pins.enb, pwm_freq)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

    def apply(self, command: WheelCommand) -> None:
        left_speed = command.left_speed
        right_speed = command.right_speed

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

        left_speed = max(-100.0, min(100.0, left_speed))
        right_speed = max(-100.0, min(100.0, right_speed))

        # Fix reversed motor wiring: flip polarity so the wheel turns the way the
        # command (and the on-screen suggestion) means it to.
        if self.cal.invert_left:
            left_speed = -left_speed
        if self.cal.invert_right:
            right_speed = -right_speed

        # Motors plugged into each other's channels -> send each speed to the
        # other side so LEFT/RIGHT turns match the command.
        if self.cal.swap_left_right:
            left_speed, right_speed = right_speed, left_speed

        self._set_left(left_speed)
        self._set_right(right_speed)

    def stop(self) -> None:
        GPIO.output([self.pins.in1, self.pins.in2, self.pins.in3, self.pins.in4], GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)

    def close(self) -> None:
        self.stop()
        self.pwm_a.stop()
        self.pwm_b.stop()
        GPIO.cleanup()

    def _set_left(self, speed: float) -> None:
        if speed > 0:
            GPIO.output(self.pins.in1, 1)
            GPIO.output(self.pins.in2, 0)
            self.pwm_a.ChangeDutyCycle(abs(speed))
        elif speed < 0:
            GPIO.output(self.pins.in1, 0)
            GPIO.output(self.pins.in2, 1)
            self.pwm_a.ChangeDutyCycle(abs(speed))
        else:
            GPIO.output(self.pins.in1, 0)
            GPIO.output(self.pins.in2, 0)
            self.pwm_a.ChangeDutyCycle(0)

    def _set_right(self, speed: float) -> None:
        if speed > 0:
            GPIO.output(self.pins.in3, 1)
            GPIO.output(self.pins.in4, 0)
            self.pwm_b.ChangeDutyCycle(abs(speed))
        elif speed < 0:
            GPIO.output(self.pins.in3, 0)
            GPIO.output(self.pins.in4, 1)
            self.pwm_b.ChangeDutyCycle(abs(speed))
        else:
            GPIO.output(self.pins.in3, 0)
            GPIO.output(self.pins.in4, 0)
            self.pwm_b.ChangeDutyCycle(0)
