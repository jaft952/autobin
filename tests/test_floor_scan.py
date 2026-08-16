"""
tests/test_floor_scan.py

Floor-scanning integration test: zigzag patrol (Layer 1) + ultrasonic
(SensorHub) + emergency stop (Layer 5), arbitrated and executed on the base.

This is the full subsumption loop:

    SensorHub.update()                                  (poll hardware)
      -> each layer .evaluate(sensors) -> Arbitrator    (vote)
      -> Arbitrator.get_winning_action()                (highest layer wins)
      -> MotionExecutor.execute(winning)                (wheels)

Run test_ultrasonic.py FIRST to confirm the sensor wiring.

Usage (Pi):
    python tests/test_floor_scan.py --no-motors    # 1st: watch decisions only,
                                                   #      wave hand at sensor
    python tests/test_floor_scan.py                # 2nd: WHEELS LIVE, zigzag
                                                   #      patrol (no camera)
    python tests/test_floor_scan.py --camera       # 3rd: + YOLO; scan pauses
                                                   #      when a can is seen

Expected behavior (wheels live, no camera):
    Robot drives a straight lane. When the wall closes to ~35 cm it pivots
    ~90 deg, hops one robot-width sideways, pivots ~90 deg again, and drives
    the return lane; the pivot side alternates each wall. Anything closer
    than ~10 cm trips Layer 5 and halts the base until cleared.

Calibrating the pattern: pass --turn-90 / --shift / --speed / --turn-speed to
seed starting values (they call ScanAroundLayer.set_timing/set_speeds); once
a set works, write it into src/subsumption/layers/layer1_scan.py as the new
default.

Live speed adjustment: while the loop is running, type into the same
terminal and press Enter:
    s 0.5       set lane cruising speed to 0.5
    t 0.4       set turn/pivot speed to 0.4
    (any other input just gets ignored)
This runs in a background thread and calls ScanAroundLayer.set_speeds(), the
same setter the dashboard uses -- just plain attribute writes, safe to call
from another thread mid-tick.

Ctrl+C stops the motors and releases GPIO.
"""
import argparse
import os
import sys
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.actuators.print_actuator import PrintActuator
from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor, UltrasonicPins
from src.hardware.sensors.sensor_hub import SensorHub
from src.subsumption.arbitrator import Arbitrator
from src.subsumption.motion_executor import MotionExecutor
from src.subsumption.layers.layer0_idle import SystemIdleLayer
from src.subsumption.layers.layer1_scan import ScanAroundLayer
from src.subsumption.layers.layer5_emergency import EmergencyStopLayer


def build_sensors(with_camera: bool) -> SensorHub:
    camera = None
    if with_camera:
        from src.hardware.sensors.camera_sensor import CameraSensor
        camera = CameraSensor()
    return SensorHub(
        front=UltrasonicSensor(UltrasonicPins(trig=23, echo=24)),
        back=UltrasonicSensor(UltrasonicPins(trig=17, echo=20)),
        front_left=UltrasonicSensor(UltrasonicPins(trig=27, echo=22)),
        front_right=UltrasonicSensor(UltrasonicPins(trig=5, echo=6)),
        camera=camera,
    )


def _speed_input_loop(scan: ScanAroundLayer, emergency: EmergencyStopLayer) -> None:
    """Background thread: 's <val>' sets lane speed, 't <val>' sets turn
    speed on both layers (layer 5 outvotes the scan pivot, so they must
    match). Runs until stdin closes."""
    for line in sys.stdin:
        parts = line.split()
        if len(parts) != 2 or parts[0] not in ("s", "t"):
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if parts[0] == "s":
            scan.set_speeds(forward_speed=value)
            print(f"[speed] lane speed -> {value}")
        else:
            scan.set_speeds(turn_speed=value)
            emergency.set_turn_speed(value)
            print(f"[speed] turn speed -> {value} (scan + emergency)")


def main():
    ap = argparse.ArgumentParser(description="Zigzag floor scan + ultrasonic test")
    ap.add_argument("--no-motors", action="store_true",
                    help="print wheel commands instead of driving")
    ap.add_argument("--camera", action="store_true",
                    help="also run YOLO litter detection (scan yields to it)")
    ap.add_argument("--hz", type=float, default=10.0, help="control loop rate")
    ap.add_argument("--turn-90", type=float, default=None,
                    help="seconds per ~90 deg pivot (calibration)")
    ap.add_argument("--shift", type=float, default=None,
                    help="seconds of the lane-spacing hop (calibration)")
    ap.add_argument("--max-lane", type=float, default=None,
                    help="seconds of straight lane before turning anyway")
    ap.add_argument("--speed", type=float, default=None,
                    help="lane cruising fraction 0..1")
    ap.add_argument("--turn-speed", type=float, default=None,
                    help="pivot fraction 0..1 (too low and the base stalls)")
    args = ap.parse_args()

    sensors = build_sensors(args.camera)
    scan = ScanAroundLayer()
    scan.set_timing(args.turn_90, args.shift, args.max_lane)
    scan.set_speeds(args.speed, args.turn_speed)
    emergency = EmergencyStopLayer()
    emergency.set_turn_speed(args.turn_speed)
    layers = [SystemIdleLayer(), scan, emergency]
    arbitrator = Arbitrator()
    executor = MotionExecutor(actuator=PrintActuator() if args.no_motors else None) # type: ignore

    period = 1.0 / args.hz
    mode = "PRINT-ONLY" if args.no_motors else "WHEELS LIVE"
    print(f"floor scan @ {args.hz:.0f} Hz — {mode} — Ctrl+C to stop")
    print(f"scan: {scan.timing_summary()}")
    print("live speed control: type 's <0..1>' or 't <0..1>' + Enter")

    threading.Thread(target=_speed_input_loop, args=(scan, emergency), daemon=True).start()

    sensors.start()
    last_msg = None
    try:
        while True:
            tick = time.monotonic()

            sensors.update()
            for layer in layers:
                arbitrator.submit_command(layer.evaluate(sensors))
            winning = arbitrator.get_winning_action()
            executor.execute(winning)
            arbitrator.clear()

            # Only print when the situation changes so the log stays readable.
            msg = f"[L{winning.layer_id}] {winning.message}"
            if msg != last_msg:
                print(msg)
                last_msg = msg

            sleep_left = period - (time.monotonic() - tick)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        executor.close()
        sensors.stop()


if __name__ == "__main__":
    main()
