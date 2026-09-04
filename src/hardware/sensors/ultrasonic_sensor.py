"""Low-level HC-SR04 driver: GPIO trigger/echo -> distance. No robot logic here."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional
import time

from src.hardware import pigpio_link

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


SPEED_OF_SOUND_CM_PER_S = 34300.0
MIN_VALID_DISTANCE_CM = 2.0

# Longest echo worth waiting for. 0.010 s of pulse is ~170 cm of range --
# well past anything this robot acts on, and a third of the 0.03 s this used
# to burn on every missing echo.
ECHO_TIMEOUT_S = 0.010

# Median of the last N pings. A single dropped echo (common on HC-SR04) used
# to swing the reported distance by tens of cm, which downstream thresholds
# read as the obstacle appearing and vanishing every tick.
MEDIAN_WINDOW = 3


@dataclass(frozen=True)
class UltrasonicPins:
    trig: int
    echo: int


class _PigpioPing:
    """Echo timing inside pigpiod. Edges are timestamped by the daemon, so a
    measurement cannot be corrupted by YOLO holding the GIL mid-pulse -- the
    failure the RPi.GPIO busy-wait below cannot defend against."""

    _WAIT_SLACK_S = 0.005   # daemon round-trip on top of the echo itself

    def __init__(self, pi, pins: UltrasonicPins) -> None:
        import pigpio  # available whenever pigpio_link.connect() succeeded

        self._pg = pigpio
        self._pi = pi
        self._pins = pins
        self._rise_tick: Optional[int] = None
        self._pulse_s: Optional[float] = None
        self._done = threading.Event()

        pi.set_mode(pins.trig, pigpio.OUTPUT)
        pi.set_mode(pins.echo, pigpio.INPUT)
        pi.write(pins.trig, 0)
        self._cb = pi.callback(pins.echo, pigpio.EITHER_EDGE, self._on_edge)

    def _on_edge(self, _gpio, level, tick) -> None:
        if level == 1:
            self._rise_tick = tick
        elif level == 0 and self._rise_tick is not None:
            self._pulse_s = self._pg.tickDiff(self._rise_tick, tick) / 1e6
            self._rise_tick = None
            self._done.set()

    def ping(self, timeout_s: float) -> Optional[float]:
        self._rise_tick = None
        self._pulse_s = None
        self._done.clear()
        self._pi.gpio_trigger(self._pins.trig, 10, 1)
        if not self._done.wait(timeout_s + self._WAIT_SLACK_S):
            return None
        return self._pulse_s

    def close(self) -> None:
        try:
            self._cb.cancel()
        finally:
            self._pi.stop()


class _GpioPing:
    """RPi.GPIO fallback: the echo is timed by a Python loop, so a GIL stall
    lands straight in the measurement. Used only without pigpiod."""

    def __init__(self, pins: UltrasonicPins) -> None:
        self._pins = pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pins.trig, GPIO.OUT)
        GPIO.setup(pins.echo, GPIO.IN)
        GPIO.output(pins.trig, False)
        time.sleep(0.05)

    def ping(self, timeout_s: float) -> Optional[float]:
        GPIO.output(self._pins.trig, True)
        time.sleep(0.00001)
        GPIO.output(self._pins.trig, False)

        timeout = time.monotonic() + timeout_s

        while GPIO.input(self._pins.echo) == 0:
            if time.monotonic() > timeout:
                return None

        pulse_start = time.monotonic()

        while GPIO.input(self._pins.echo) == 1:
            if time.monotonic() > timeout:
                return None

        return time.monotonic() - pulse_start

    def close(self) -> None:
        # Some host environments provide a GPIO module but never had
        # `setmode()` called (or it was cleaned up elsewhere). Calling
        # `GPIO.cleanup()` in that state raises a RuntimeError:
        # "Please set pin numbering mode using GPIO.setmode(...)".
        # Guard by checking the current mode first where available.
        try:
            mode = GPIO.getmode()
        except Exception:
            mode = None

        if mode is not None:
            GPIO.cleanup([self._pins.trig, self._pins.echo])


def _make_backend(pins: UltrasonicPins):
    """pigpiod if it is running, else RPi.GPIO, else nothing (dev machine)."""
    pi = pigpio_link.connect()
    if pi is not None:
        return _PigpioPing(pi, pins)
    if GPIO is not None:
        return _GpioPing(pins)
    return None


class UltrasonicSensor:

    def __init__(self, pins: UltrasonicPins):

        self._pins = pins
        self._distance_cm: Optional[float] = None
        self._updated_at: Optional[float] = None
        self._history: deque = deque(maxlen=MEDIAN_WINDOW)
        self._lock = threading.Lock()   # update() runs on UltrasonicArray's thread, get_distance_cm() on the control loop
        self._backend = _make_backend(pins)

    def update(self) -> None:
        """
        Perform one ultrasonic measurement.

        Stores the latest measured distance internally.
        """

        if self._backend is None:
            self._record(None)
            return

        pulse_s = self._backend.ping(ECHO_TIMEOUT_S)
        if pulse_s is None:
            self._record(None)
            return

        distance_cm = pulse_s * SPEED_OF_SOUND_CM_PER_S / 2.0
        self._record(distance_cm if distance_cm >= MIN_VALID_DISTANCE_CM else None)

    def _record(self, reading: Optional[float]) -> None:
        """Median filter: None (no echo) stays in the window so a dropout alone can't move the reading."""
        with self._lock:
            self._history.append(reading)
            valid = sorted(r for r in self._history if r is not None)
            # Lower of the two middles on an even count: report the nearer obstacle.
            self._distance_cm = (valid[(len(valid) - 1) // 2]
                                  if len(valid) * 2 > len(self._history) else None)
            self._updated_at = time.monotonic()

    def get_distance_cm(self) -> Optional[float]:
        with self._lock:
            return self._distance_cm

    def get_distance_age_s(self) -> Optional[float]:
        """Seconds since the reported distance was last recomputed, or None
        before the first ping. Callers gating on a threshold need this: the
        round-robin schedule means a reading can be several ticks old."""
        with self._lock:
            if self._updated_at is None:
                return None
            return time.monotonic() - self._updated_at

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
