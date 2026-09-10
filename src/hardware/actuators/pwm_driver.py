from typing import Protocol

from src.hardware import pigpio_link
from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import WheelCommand

# 200 Hz keeps software PWM duty accurate under CPU load; board max is 2 kHz.
PWM_FREQ_HZ = 200
ZKBM1_MAX_PWM_HZ = 2000

# Static friction floor: duty below this buzzes without turning. 0 = off.
MIN_MOVE_DUTY = 0.0   # TODO tune on hardware: raise until the chassis creeps


def _apply_stiction_floor(left: float, right: float):
    """Scale a too-weak wheel pair up together, keeping the left/right ratio (and steering) intact."""
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
    """One pin's PWM via pigpiod: DMA-timed, so duty stays accurate under CPU load. Same interface as RPi.GPIO's PWM."""

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


class PWMActuator:
    """Converts wheel command into GPIO + PWM signals for the ZK-BM1 dual H-bridge (no ENA/ENB; duty on the input pins is speed+direction)."""

    def __init__(self, pins: MotorPins | None = None, calibration: MotionCalibration | None = None, pwm_freq: int = PWM_FREQ_HZ) -> None:
        if pwm_freq > ZKBM1_MAX_PWM_HZ:
            raise ValueError(f"pwm_freq {pwm_freq} Hz exceeds the ZK-BM1's "
                             f"~{ZKBM1_MAX_PWM_HZ} Hz input limit")
        self.pins = pins or MotorPins()
        self.cal = calibration or MotionCalibration()
        self.last_duty = (0.0, 0.0)
        self._my_pins = [self.pins.in1, self.pins.in2, self.pins.in3, self.pins.in4]
        # What was last written to each input pin. last_duty is the wheel
        # speed the mixer asked for; this is what the H-bridge actually sees,
        # which is the only thing that says coast (all LOW) from brake (all
        # HIGH) from still driving.
        self.last_pin_duty = {pin: 0.0 for pin in self._my_pins}

        # Prefer pigpiod; fall back to RPi.GPIO soft PWM if unavailable.
        self._pi = pigpio_link.connect()
        if self._pi is not None:
            self.pwm_in1 = _PigpioPWM(self._pi, self.pins.in1, pwm_freq)
            self.pwm_in2 = _PigpioPWM(self._pi, self.pins.in2, pwm_freq)
            self.pwm_in3 = _PigpioPWM(self._pi, self.pins.in3, pwm_freq)
            self.pwm_in4 = _PigpioPWM(self._pi, self.pins.in4, pwm_freq)
            print("[PWMActuator] pigpio backend (hardware-timed PWM)")
            return

        # Release only our own pins; a global cleanup would break other modules' GPIO.
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

        # Swap first: trim/invert describe the channel, not the command, and must apply after the swap.
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

        # Flip polarity to fix reversed motor wiring.
        if self.cal.invert_left:
            left_speed = -left_speed
        if self.cal.invert_right:
            right_speed = -right_speed

        # Telemetry: final duty sent, after all corrections.
        self.last_duty = (left_speed, right_speed)
        self._set_left(left_speed)
        self._set_right(right_speed)

    def _write(self, pwm, pin: int, value: float) -> None:
        pwm.ChangeDutyCycle(value)
        self.last_pin_duty[pin] = value

    def stop(self) -> None:
        """Coast: both inputs LOW, windings open, wheels free to spin. Use brake() to hold position."""
        self.last_duty = (0.0, 0.0)
        for pwm, pin in self._channels():
            self._write(pwm, pin, 0)

    def brake(self) -> None:
        """ACTIVE BRAKE ('Fast Motor Stop'): both inputs of each motor driven
        HIGH (PWM 100%) at the same time. This shorts the motor windings, so
        any attempt to turn the wheel — a push forward or back — induces a
        current that opposes the motion (dynamic braking). The robot resists
        being rolled, holding position while the arm actuates. No mechanical
        brake exists; this is the strongest hold the hardware allows.
        Stationary, it draws ~no current; current only flows while something
        is actively trying to move it."""
        # ORDER MATTERS, same reason as _set_left(): raising the inputs in
        # pin order walks past a state where one input is already HIGH and
        # its partner still carries the old duty -- which is not a brake, it
        # is a near-full-speed DRIVE on that one channel, and only on that
        # one, so the chassis lurches and yaws before it stops. Drop
        # everything to LOW first (a harmless coast) and raise from there.
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
                self._pi.stop() # type: ignore ; disconnect from pigpiod only          
            except Exception:
                pass
            return
        # Own pins only — a global cleanup wipes the ultrasonic's setup too.
        try:
            GPIO.cleanup(self._my_pins)  # type: ignore
        except Exception:
            pass

    def _channels(self):
        return ((self.pwm_in1, self.pins.in1), (self.pwm_in2, self.pins.in2),
                (self.pwm_in3, self.pins.in3), (self.pwm_in4, self.pins.in4))

    def _set_left(self, speed: float) -> None:
        # Drop the idle input to 0 before raising the active one, or a direction change briefly brakes.
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
