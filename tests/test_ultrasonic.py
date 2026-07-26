"""
tests/test_ultrasonic.py

HC-SR04 wiring check. Run this FIRST, before test_floor_scan.py.

Usage (Pi):
    python tests/test_ultrasonic.py                 # default pins TRIG=BCM5 ECHO=BCM6
    python tests/test_ultrasonic.py --trig 5 --echo 6 --hz 10

Wiring (BCM numbering):
    VCC  -> 5V   (physical pin 2)
    GND  -> GND  (physical pin 6)
    TRIG -> GPIO 23 (physical pin 16)
    ECHO -> GPIO 24 (physical pin 18)
            ECHO outputs 5V, Pi GPIO tolerates only 3.3V!

What you should see:
    A distance readout ~10x/s with a bar that shrinks as you move your hand
    toward the sensor. "--" means no echo (nothing in range, wiring fault,
    or not running on the Pi). Check it reads sensibly at 10/30/100 cm.
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor, UltrasonicPins

BAR_FULL_CM = 100.0  # bar spans 0..1 m
BAR_WIDTH = 40


def main():
    ap = argparse.ArgumentParser(description="HC-SR04 wiring check")
    ap.add_argument("--trig", type=int, default=23, help="TRIG pin (GPIO)")
    ap.add_argument("--echo", type=int, default=24, help="ECHO pin (GPIO)")
    ap.add_argument("--hz", type=float, default=10.0, help="poll rate")
    args = ap.parse_args()

    sensor = UltrasonicSensor(UltrasonicPins(trig=args.trig, echo=args.echo))
    period = 1.0 / args.hz
    print(f"TRIG=GPIO{args.trig} ECHO=GPIO{args.echo} @ {args.hz:.0f} Hz — Ctrl+C to quit")

    try:
        while True:
            tick = time.monotonic()
            sensor.update()
            dist = sensor.get_distance_cm()
            if dist is None:
                line = "dist   --      |" + " " * BAR_WIDTH + "|"
            else:
                filled = int(min(dist, BAR_FULL_CM) / BAR_FULL_CM * BAR_WIDTH)
                line = f"dist {dist:6.1f} cm |" + "#" * filled + " " * (BAR_WIDTH - filled) + "|"
            print("\r" + line, end="", flush=True)

            sleep_left = period - (time.monotonic() - tick)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
