"""
tests/test_turn_direction.py

Turn-direction bench test. Nothing but pivots, so a mirrored turn can be
traced to the ONE stage that caused it.

Reported symptom: covering the RIGHT diagonal logs "turn left" and the wheels
turn right; covering the LEFT one is mirrored too. Forward drive is correct.
Two different faults look identical in the floor-scan log:

    A. the two diagonal ultrasonics are swapped (front_left reads the right
       side), so the software picks the wrong side and the wheels obey it
    B. the two motors are plugged into each other's ZK-BM1 channels, so the
       software picks the right side and the wheels do the opposite

MANUAL mode tells them apart: it commands a turn with no sensor involved.
    - manual "l" turns the chassis RIGHT  -> fault B (wiring)
    - manual "l" turns the chassis LEFT   -> wheels are fine, so fault A

Usage (Pi):
    python tests/test_turn_direction.py --manual              # no sensors
    python tests/test_turn_direction.py --manual --no-motors  # dry run first
    python tests/test_turn_direction.py                       # sensor-driven
    python tests/test_turn_direction.py --speed 0.4

MANUAL mode: type into the same terminal and press Enter.
    l   pivot left  (v_theta positive = CCW)
    r   pivot right
    s   stop
Each command runs for --burst seconds, then stops on its own.

SENSOR mode: cover one diagonal ultrasonic with your hand. The robot pivots
AWAY from the covered side, printing the reading, the chosen side and the
duty that reached each ZK-BM1 input pin. Uncover it and the wheels stop.

Both modes brake between opposite directions (never flip a running motor) and
release GPIO on Ctrl+C.
"""
import argparse
import os
import sys
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.actuators.print_actuator import PrintActuator
from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor, UltrasonicPins
from src.motion.calibration import MotionCalibration
from src.subsumption.arbitrator import ActionCommand
from src.subsumption.motion_executor import MotionExecutor

COVERED_CM = 25.0    # a hand over the sensor reads well inside this
SETTLE_S = 0.2       # stop between opposite directions (safety doc rule 7)


def pin_duty(executor: MotionExecutor):
    """What reached the four ZK-BM1 inputs, or None if the actuator has no
    last_duty. Derived, not read back: RPi.GPIO.PWM cannot report its duty."""
    duty = getattr(executor.actuator, "last_duty", None)
    if duty is None:
        return None
    left, right = duty
    pins = (max(left, 0.0), max(-left, 0.0), max(right, 0.0), max(-right, 0.0))
    return "IN1 %5.1f  IN2 %5.1f  IN3 %5.1f  IN4 %5.1f" % pins


class Pivot:
    """Drives one pivot at a time, braking before the direction flips."""

    def __init__(self, executor: MotionExecutor, speed: float):
        self.executor = executor
        self.speed = speed
        self._dir = 0.0

    def set(self, direction: float) -> None:
        """direction: +1 left (CCW), -1 right, 0 stop."""
        if direction == self._dir:
            return
        if self._dir != 0.0 and direction != 0.0:
            self.executor.stop()
            time.sleep(SETTLE_S)
        self._dir = direction
        if direction == 0.0:
            self.executor.stop()
            return
        self.executor.execute(ActionCommand(
            layer_id=1, active=True,
            motion_vector=(0, 0, direction * self.speed),
            arm_action='stop', message="pivot",
        ))

    def stop(self) -> None:
        self.set(0.0)


def report(executor, wanted: str, v_theta: float, extra: str = "") -> None:
    line = f"[cmd] {wanted:5s}  v_theta {v_theta:+.2f}"
    if extra:
        line += f"  {extra}"
    duty = pin_duty(executor)
    if duty:
        line += f"\n      {duty}"
    print(line)


def run_manual(executor: MotionExecutor, speed: float, burst_s: float) -> None:
    pivot = Pivot(executor, speed)
    pending = []
    lock = threading.Lock()

    def reader():
        for line in sys.stdin:
            key = line.strip().lower()
            if key in ("l", "r", "s", "q"):
                with lock:
                    pending.append(key)

    threading.Thread(target=reader, daemon=True).start()
    print("manual pivot - type l / r / s + Enter (q quits)")
    print(f"each burst runs {burst_s:.1f}s then stops")

    ends_at = 0.0
    while True:
        with lock:
            key = pending.pop(0) if pending else None
        if key == "q":
            return
        if key in ("l", "r"):
            direction = 1.0 if key == "l" else -1.0
            pivot.set(direction)
            report(executor, "LEFT" if key == "l" else "RIGHT", direction * speed,
                   "watch the chassis, not the log")
            ends_at = time.monotonic() + burst_s
        elif key == "s":
            pivot.stop()
            ends_at = 0.0
        if ends_at and time.monotonic() >= ends_at:
            pivot.stop()
            ends_at = 0.0
        time.sleep(0.05)


def run_sensor(executor: MotionExecutor, speed: float, hz: float) -> None:
    left = UltrasonicSensor(UltrasonicPins(trig=27, echo=22))
    right = UltrasonicSensor(UltrasonicPins(trig=5, echo=6))
    pivot = Pivot(executor, speed)
    period = 1.0 / hz
    sensors = (left, right)
    turn = 0
    last = None

    print("sensor pivot - cover ONE diagonal with your hand")
    print("front_left = BCM 27/22, front_right = BCM 5/6")
    try:
        while True:
            tick = time.monotonic()
            sensors[turn].update()
            turn = (turn + 1) % len(sensors)

            l = left.get_distance_cm()
            r = right.get_distance_cm()
            l_covered = l is not None and l < COVERED_CM
            r_covered = r is not None and r < COVERED_CM

            if l_covered and not r_covered:
                direction, wanted = -1.0, "RIGHT"
            elif r_covered and not l_covered:
                direction, wanted = 1.0, "LEFT"
            else:
                direction, wanted = 0.0, "STOP"

            pivot.set(direction)
            state = (wanted, l_covered, r_covered)
            if state != last:
                lt = "--" if l is None else f"{l:.0f}cm"
                rt = "--" if r is None else f"{r:.0f}cm"
                report(executor, wanted, direction * speed,
                       f"front_left {lt}  front_right {rt}")
                last = state

            sleep_left = period - (time.monotonic() - tick)
            if sleep_left > 0:
                time.sleep(sleep_left)
    finally:
        pivot.stop()
        for s in sensors:
            s.close()


def main():
    ap = argparse.ArgumentParser(description="Turn-direction bench test")
    ap.add_argument("--manual", action="store_true",
                    help="keyboard pivots, no sensors (isolates the wiring)")
    ap.add_argument("--no-motors", action="store_true",
                    help="print wheel commands instead of driving")
    ap.add_argument("--speed", type=float, default=0.32,
                    help="pivot fraction 0..1")
    ap.add_argument("--burst", type=float, default=1.5,
                    help="manual mode: seconds per pivot")
    ap.add_argument("--hz", type=float, default=20.0, help="sensor loop rate")
    args = ap.parse_args()

    speed = max(0.0, min(1.0, args.speed))
    executor = MotionExecutor(actuator=PrintActuator() if args.no_motors else None)  # type: ignore

    cal = MotionCalibration()
    print(f"invert_left={cal.invert_left} invert_right={cal.invert_right} "
          f"swap_left_right={cal.swap_left_right}")
    print("v_theta positive = CCW = LEFT")

    try:
        if args.manual:
            run_manual(executor, speed, args.burst)
        else:
            run_sensor(executor, speed, args.hz)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        executor.close()


if __name__ == "__main__":
    main()
