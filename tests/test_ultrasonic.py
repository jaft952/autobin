"""
tests/test_ultrasonic.py

HC-SR04 wiring check — TOP + BOTTOM sensors. Run this FIRST, before
test_floor_scan.py.

Usage (Pi):
    python tests/test_ultrasonic.py                 # default pins, see below
    python tests/test_ultrasonic.py --trig2 27 --echo2 22 --hz 10

Wiring (BCM numbering):
    TOP sensor:
        VCC  -> 5V   (physical pin 2)
        GND  -> GND  (physical pin 6)
        TRIG -> GPIO 23 (physical pin 16)
        ECHO -> GPIO 24 (physical pin 18)
    BOTTOM sensor:
        VCC  -> 5V   (physical pin 4)
        GND  -> GND  (physical pin 9)
        TRIG -> GPIO 27 (physical pin 13)
        ECHO -> GPIO 22 (physical pin 15)
                ECHO outputs 5V, Pi GPIO tolerates only 3.3V!

What you should see:
    Two distance readouts side by side (top/bottom) ~10x/s, each with a bar
    that shrinks as you move your hand toward that sensor. "--" means no
    echo (nothing in range, wiring fault, or not running on the Pi). Check
    both read sensibly at 10/30/100 cm.
"""
import argparse
import os
import sys
import time
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor, UltrasonicPins

BAR_FULL_CM = 100.0  # bar spans 0..1 m
BAR_WIDTH = 40


def _reading_str(label: str, dist: Optional[float]) -> str:
    val = f"{'--':>9s}" if dist is None else f"{dist:6.1f} cm"
    filled = 0 if dist is None else int(min(dist, BAR_FULL_CM) / BAR_FULL_CM * BAR_WIDTH)
    bar = "#" * filled + " " * (BAR_WIDTH - filled)
    return f"{label:<6s} {val} |{bar}|"


def main():
    ap = argparse.ArgumentParser(description="HC-SR04 wiring check (top + bottom)")
    ap.add_argument("--trig", type=int, default=23, help="top sensor TRIG pin (GPIO)")
    ap.add_argument("--echo", type=int, default=24, help="top sensor ECHO pin (GPIO)")
    ap.add_argument("--trig2", type=int, default=27, help="bottom sensor TRIG pin (GPIO)")
    ap.add_argument("--echo2", type=int, default=22, help="bottom sensor ECHO pin (GPIO)")
    ap.add_argument("--hz", type=float, default=10.0, help="poll rate")
    args = ap.parse_args()

    sensor_top = UltrasonicSensor(UltrasonicPins(trig=args.trig, echo=args.echo))
    sensor_bottom = UltrasonicSensor(UltrasonicPins(trig=args.trig2, echo=args.echo2))
    period = 1.0 / args.hz
    print(f"top TRIG=GPIO{args.trig} ECHO=GPIO{args.echo}  |  "
          f"bottom TRIG=GPIO{args.trig2} ECHO=GPIO{args.echo2} "
          f"@ {args.hz:.0f} Hz — Ctrl+C to quit")

    try:
        while True:
            tick = time.monotonic()
            sensor_top.update()
            sensor_bottom.update()
            line = (_reading_str("top", sensor_top.get_distance_cm()) + "    " +
                    _reading_str("bottom", sensor_bottom.get_distance_cm()))
            print("\r" + line, end="", flush=True)

            sleep_left = period - (time.monotonic() - tick)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        sensor_top.close()
        sensor_bottom.close()


if __name__ == "__main__":
    main()
