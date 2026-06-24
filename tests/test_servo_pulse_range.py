# tests/test_servo_pulse_range.py
"""
Find the REAL travel of one YF-6125MG channel by sweeping raw pulse widths.

This bypasses the .angle abstraction and commands microseconds directly, so you
can watch exactly where the servo starts moving, where it stops, and where it
stalls against a hard stop (buzzing / not tracking).

Run on the Pi:  python -m tests.test_servo_pulse_range
Watch the joint and note the pulse values where motion begins and ends.
EDIT `CHANNEL` to the joint you want to test. Keep a hand near the power switch.
"""
import time
from adafruit_servokit import ServoKit

CHANNEL = 0          # which servo channel to sweep (0 == CH1)
PULSE_START_US = 500 # begin inside the known-safe band
PULSE_END_US = 2500  # end inside the known-safe band
STEP_US = 50         # resolution of the sweep
DWELL_S = 0.4        # pause at each step so you can see the position

# To probe BEYOND the safe band, widen these CAUTIOUSLY (e.g. 400 / 2600) and
# stop immediately if the servo buzzes/stalls instead of moving — that is the
# hard stop, not extra usable range.

kit = ServoKit(channels=16)
servo = kit.servo[CHANNEL]

# Use the widest window we intend to probe so set_pulse_width_range never clips.
servo.set_pulse_width_range(PULSE_START_US, PULSE_END_US)
servo.actuation_range = 180  # mapping is irrelevant here; we drive pulses directly


def write_pulse_us(s, pulse_us):
    """Command a raw pulse width (µs) regardless of actuation_range, via the
    underlying PWM channel. 50 Hz => 20000 µs period."""
    period_us = 1_000_000 / s._pwm_out.frequency
    duty = int(pulse_us / period_us * 0xFFFF)
    s._pwm_out.duty_cycle = max(0, min(0xFFFF, duty))


print(f"Sweeping CH{CHANNEL + 1} from {PULSE_START_US} to {PULSE_END_US} µs")
print("Watch the joint. Note where it starts/stops tracking.\n")

for pulse in range(PULSE_START_US, PULSE_END_US + 1, STEP_US):
    write_pulse_us(servo, pulse)
    print(f"  CH{CHANNEL + 1}: {pulse} µs")
    time.sleep(DWELL_S)

print("\nSweep up done. Sweeping back down...")
for pulse in range(PULSE_END_US, PULSE_START_US - 1, -STEP_US):
    write_pulse_us(servo, pulse)
    time.sleep(DWELL_S)

print("Done. The pulses where it actually moved = the usable range for this servo.")
