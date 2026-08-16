"""
tests/test_ultrasonic.py

HC-SR04 wiring check -- up to 4 sensors. Run this FIRST, before
test_floor_scan.py.

Usage (Pi):
    python tests/test_ultrasonic.py                 # default pins, see below
    python tests/test_ultrasonic.py --trig3 5 --echo3 6 --hz 10

Wiring (BCM numbering):
    front:
        TRIG -> GPIO 23 (physical pin 16)
        ECHO -> GPIO 24 (physical pin 18)
    front_left (diagonal):
        TRIG -> GPIO 27 (physical pin 13)
        ECHO -> GPIO 22 (physical pin 15)
    front_right (diagonal):
        TRIG -> GPIO 5  (physical pin 29)
        ECHO -> GPIO 6  (physical pin 31)
    back:
        TRIG -> GPIO 17 (physical pin 11)
        ECHO -> GPIO 20 (physical pin 38)
    All sensors: VCC -> 5V, GND -> GND (shared rail is fine).
                 ECHO outputs 5V, Pi GPIO tolerates only 3.3V --
                 use a voltage divider on every ECHO line!

What you should see:
    One distance readout per sensor, ~10x/s, each with a bar that shrinks
    as you move your hand toward that sensor. "--" means no echo (nothing
    in range, wiring fault, or not running on the Pi). Check each reads
    sensibly at 10/30/100 cm.
"""
import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor, UltrasonicPins

BAR_FULL_CM = 100.0  # bar spans 0..1 m
BAR_WIDTH = 40

DEFAULT_SENSORS: List[Tuple[str, int, int]] = [
    ("front", 23, 24),
    ("front_left", 27, 22),
    ("front_right", 5, 6),
    ("back", 17, 20),
]


def _reading_str(label: str, dist: Optional[float]) -> str:
    val = f"{'--':>9s}" if dist is None else f"{dist:6.1f} cm"
    filled = 0 if dist is None else int(min(dist, BAR_FULL_CM) / BAR_FULL_CM * BAR_WIDTH)
    bar = "#" * filled + " " * (BAR_WIDTH - filled)
    return f"{label:<8s} {val} |{bar}|"


def main():
    ap = argparse.ArgumentParser(description="HC-SR04 wiring check (up to 4 sensors)")
    for label, default_trig, default_echo in DEFAULT_SENSORS:
        ap.add_argument(f"--trig-{label}", dest=f"trig_{label}", type=int,
                         default=default_trig, help=f"{label} sensor TRIG pin (GPIO)")
        ap.add_argument(f"--echo-{label}", dest=f"echo_{label}", type=int,
                         default=default_echo, help=f"{label} sensor ECHO pin (GPIO)")
    ap.add_argument("--hz", type=float, default=10.0, help="poll rate")
    args = ap.parse_args()

    sensors = []
    for label, _, _ in DEFAULT_SENSORS:
        trig = getattr(args, f"trig_{label}")
        echo = getattr(args, f"echo_{label}")
        sensors.append((label, UltrasonicSensor(UltrasonicPins(trig=trig, echo=echo))))
        print(f"{label:<8s} TRIG=GPIO{trig} ECHO=GPIO{echo}")

    period = 1.0 / args.hz
    print(f"@ {args.hz:.0f} Hz -- Ctrl+C to quit")

    first = True
    try:
        while True:
            tick = time.monotonic()
            for _, sensor in sensors:
                sensor.update()
            lines = [_reading_str(label, sensor.get_distance_cm()) for label, sensor in sensors]

            if first:
                print("\n".join(lines), end="", flush=True)
                first = False
            else:
                print(f"\x1b[{len(lines) - 1}A\r" + "\n".join(lines), end="", flush=True)

            sleep_left = period - (time.monotonic() - tick)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        for _, sensor in sensors:
            sensor.close()


if __name__ == "__main__":
    main()
