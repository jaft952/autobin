"""Minimal TB6612FNG test. No keyboard needed.

Drives the LEFT motor forward for 3 seconds, then the RIGHT motor,
printing each step so you can see exactly where it stops.

Run with:
    python3 minimal_test.py
"""

# pyright: reportAttributeAccessIssue=false

import time

try:
    import RPi.GPIO as GPIO
    print("RPi.GPIO imported OK. Version:", GPIO.VERSION)
except Exception as error:
    print("Could not import RPi.GPIO:", error)
    raise SystemExit

# Standard pins
STBY = 25
AIN1, AIN2, PWMA = 24, 23, 18
BIN1, BIN2, PWMB = 17, 27, 13

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in (STBY, AIN1, AIN2, PWMA, BIN1, BIN2, PWMB):
    GPIO.setup(pin, GPIO.OUT)

pwma = GPIO.PWM(PWMA, 1000)
pwmb = GPIO.PWM(PWMB, 1000)
pwma.start(0)
pwmb.start(0)

try:
    print("Setting STBY high")
    GPIO.output(STBY, GPIO.HIGH)
    time.sleep(0.5)

    print("Left motor forward at full speed for 3 seconds")
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)
    pwma.ChangeDutyCycle(100)
    time.sleep(3)
    pwma.ChangeDutyCycle(0)

    print("Right motor forward at full speed for 3 seconds")
    GPIO.output(BIN1, GPIO.HIGH)
    GPIO.output(BIN2, GPIO.LOW)
    pwmb.ChangeDutyCycle(100)
    time.sleep(3)
    pwmb.ChangeDutyCycle(0)

    print("Done")
finally:
    pwma.stop()
    pwmb.stop()
    GPIO.cleanup()
    print("Cleaned up")