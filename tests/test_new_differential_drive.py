"""All in one keyboard test for the TB6612FNG differential drive base.

Run with:
    python3 motor_test.py

Controls:
    w = forward      s = backward
    a = spin left    d = spin right
    space or x = stop
    k = faster       j = slower
    q = quit

Lift the wheels off the ground before running.
Ctrl C works as an emergency stop at any time.
"""

# termios and tty are Linux only, so Pylance on Windows wrongly flags
# their functions as unknown. This line silences that false warning.
# The code still runs fine on the Raspberry Pi.
# pyright: reportAttributeAccessIssue=false

import sys
import tty
import termios
import select

import RPi.GPIO as GPIO

# ---- Pin assignments (BCM numbering) ----
# Left motor (channel A)
AIN1_PIN = 24
AIN2_PIN = 23
PWMA_PIN = 18

# Right motor (channel B)
BIN1_PIN = 17
BIN2_PIN = 27
PWMB_PIN = 13

# STBY enable pin. The TB6612FNG only runs when STBY is HIGH.
# If you wired STBY straight to 3.3V, set STBY_PIN = None to skip it.
STBY_PIN = 25

# ---- PWM and movement settings ----
PWM_FREQUENCY_HZ = 1000
MAX_DUTY = 100

START_SPEED = 60
MIN_SPEED = 20
MAX_SPEED = 100
SPEED_STEP = 10
LOOP_DELAY = 0.05

# ---- Direction correction ----
# Set to True if a wheel spins the opposite way to what you expect.
LEFT_INVERTED = False
RIGHT_INVERTED = False

HELP_TEXT = """
Controls:
  w = forward      s = backward
  a = spin left    d = spin right
  space / x = stop
  k = faster       j = slower
  q = quit
"""


class Motor:
    """One H bridge channel of the TB6612FNG."""

    def __init__(self, in1, in2, pwm_pin, inverted=False):
        self._in1 = in1
        self._in2 = in2
        self._inverted = inverted

        GPIO.setup(in1, GPIO.OUT)
        GPIO.setup(in2, GPIO.OUT)
        GPIO.setup(pwm_pin, GPIO.OUT)

        self._pwm = GPIO.PWM(pwm_pin, PWM_FREQUENCY_HZ)
        self._pwm.start(0)

    def drive(self, speed):
        """Drive the motor. speed is -100 to 100. Positive is forward."""
        if self._inverted:
            speed = -speed
        speed = max(-MAX_DUTY, min(MAX_DUTY, speed))

        if speed > 0:
            GPIO.output(self._in1, GPIO.HIGH)
            GPIO.output(self._in2, GPIO.LOW)
        elif speed < 0:
            GPIO.output(self._in1, GPIO.LOW)
            GPIO.output(self._in2, GPIO.HIGH)
        else:
            GPIO.output(self._in1, GPIO.LOW)
            GPIO.output(self._in2, GPIO.LOW)

        self._pwm.ChangeDutyCycle(abs(speed))

    def stop(self):
        self.drive(0)

    def cleanup(self):
        self._pwm.stop()


class DifferentialDrive:
    """Two motors wired to a TB6612FNG, driven as a differential base."""

    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        if STBY_PIN is not None:
            GPIO.setup(STBY_PIN, GPIO.OUT)

        self.left = Motor(AIN1_PIN, AIN2_PIN, PWMA_PIN, LEFT_INVERTED)
        self.right = Motor(BIN1_PIN, BIN2_PIN, PWMB_PIN, RIGHT_INVERTED)

        self.enable()

    def enable(self):
        if STBY_PIN is not None:
            GPIO.output(STBY_PIN, GPIO.HIGH)

    def disable(self):
        if STBY_PIN is not None:
            GPIO.output(STBY_PIN, GPIO.LOW)

    def drive(self, left_speed, right_speed):
        self.left.drive(left_speed)
        self.right.drive(right_speed)

    def forward(self, speed):
        self.drive(speed, speed)

    def backward(self, speed):
        self.drive(-speed, -speed)

    def turn_left(self, speed):
        self.drive(-speed, speed)

    def turn_right(self, speed):
        self.drive(speed, -speed)

    def stop(self):
        self.drive(0, 0)

    def cleanup(self):
        self.stop()
        self.disable()
        self.left.cleanup()
        self.right.cleanup()
        GPIO.cleanup()


class KeyPoller:
    """Read one key at a time without waiting for Enter. Works over SSH."""

    def __init__(self, stream=None):
        self._stream = stream or sys.stdin
        self._fd = self._stream.fileno()
        self._old_settings = None

    def __enter__(self):
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def poll(self, timeout=0.0):
        ready, _, _ = select.select([self._stream], [], [], timeout)
        if ready:
            return self._stream.read(1)
        return None


def apply_action(robot, action, speed):
    if action == "forward":
        robot.forward(speed)
    elif action == "backward":
        robot.backward(speed)
    elif action == "left":
        robot.turn_left(speed)
    elif action == "right":
        robot.turn_right(speed)
    else:
        robot.stop()


def main():
    robot = DifferentialDrive()
    speed = START_SPEED
    action = "stop"

    print(HELP_TEXT)
    print("Lift the wheels off the ground. Ready.")

    try:
        with KeyPoller() as keys:
            while True:
                key = keys.poll(LOOP_DELAY)
                if key is None:
                    continue

                key = key.lower()

                if key == "q":
                    break
                elif key == "w":
                    action = "forward"
                elif key == "s":
                    action = "backward"
                elif key == "a":
                    action = "left"
                elif key == "d":
                    action = "right"
                elif key in (" ", "x"):
                    action = "stop"
                elif key == "k":
                    speed = min(MAX_SPEED, speed + SPEED_STEP)
                elif key == "j":
                    speed = max(MIN_SPEED, speed - SPEED_STEP)
                else:
                    continue

                apply_action(robot, action, speed)
                print(f"action = {action:<9} speed = {speed:>3}   ", end="\r")

    except KeyboardInterrupt:
        pass
    finally:
        robot.cleanup()
        print("\nStopped and cleaned up.")


if __name__ == "__main__":
    main()