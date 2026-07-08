"""
src/hardware/sensors/ultrasonic_sensor.py

HC-SR04 ultrasonic distance driver (front-facing obstacle ranging).

WIRING (BCM numbering, defaults below):
    VCC  -> 5V        (physical pin 2)
    GND  -> GND       (physical pin 6)
    TRIG -> BCM 5     (physical pin 29)  - 3.3V trigger is fine for the HC-SR04
    ECHO -> BCM 6     (physical pin 31)  - !! THROUGH A VOLTAGE DIVIDER !!

    The ECHO pin outputs 5V but Pi GPIO is 3.3V-only. Divide it down, e.g.:
        ECHO --[1k]--+--> BCM 6
                     |
                    [2k]
                     |
                    GND

NOTE: BCM 5/6 were chosen because the L298N motor driver already occupies
17/27/22/23/18/19 and the PCA9685 uses I2C (BCM 2/3). The old config default
(trig=23) collided with motor IN4 — do not wire it there.

DESIGN (matches the polled-sensor abstraction, see docs/subsumption_constraints.md):
    update()             — called once per main-loop tick; fires ONE ping and
                           stores the result. One ping keeps the blocking time
                           bounded (~50 ms worst case on timeout, ~2 ms when an
                           object is nearby), which fits a 10 Hz tick and also
                           respects the HC-SR04's recommended 60 ms cycle.
    get_distance_cm()    — median of the last few valid pings (spike filter),
                           or None if we have no fresh reading (sensor unplugged,
                           target out of range, or running on a dev machine).

On a non-Pi machine (no RPi.GPIO) the driver silently degrades: every reading
is None, so callers must treat None as "no obstacle information", never as 0.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Optional

try:
    import RPi.GPIO as GPIO  # type: ignore
    _ON_PI = True
except Exception:  # pragma: no cover - dev machine fallback
    GPIO = None
    _ON_PI = False


@dataclass(frozen=True)
class UltrasonicPins:
    trig: int = 5
    echo: int = 6


# Speed of sound at ~20C is ~343 m/s; echo time is out-and-back, so:
#   distance_cm = high_time_s * 34300 / 2
_CM_PER_SECOND_HALVED = 34300.0 / 2.0

# How long to wait for the echo line to rise after triggering, and how long a
# high pulse we accept before declaring "nothing in range". 30 ms of high time
# corresponds to ~5 m, past the HC-SR04's usable range anyway.
_ECHO_RISE_TIMEOUT_S = 0.020
_ECHO_FALL_TIMEOUT_S = 0.030


class UltrasonicSensor:
    """Polled HC-SR04 driver. update() once per tick, get_distance_cm() any time."""

    def __init__(
        self,
        pins: UltrasonicPins | None = None,
        min_range_cm: float = 2.0,
        max_range_cm: float = 300.0,
        window: int = 3,
        stale_after_s: float = 0.6,
    ) -> None:
        self.pins = pins or UltrasonicPins()
        self.min_range_cm = min_range_cm
        self.max_range_cm = max_range_cm
        self.stale_after_s = stale_after_s

        # Rolling buffer of recent VALID readings for the median filter.
        self._readings: deque[float] = deque(maxlen=window)
        self._last_valid_at: float = 0.0

        if _ON_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pins.trig, GPIO.OUT)
            GPIO.setup(self.pins.echo, GPIO.IN)
            GPIO.output(self.pins.trig, GPIO.LOW)
            time.sleep(0.05)  # let the module settle after power-up

    # ── Polled abstraction ────────────────────────────────────────────────

    def update(self) -> None:
        """Fire one ping and record it. Called every tick by the main loop."""
        cm = self._ping_cm()
        if cm is not None:
            self._readings.append(cm)
            self._last_valid_at = time.monotonic()

    def get_distance_cm(self) -> Optional[float]:
        """Median-filtered distance in cm, or None when there is no fresh data.

        None means "don't know" (out of range / not on Pi / sensor fault) —
        callers must NOT treat it as an obstacle at 0 cm.
        """
        if not self._readings:
            return None
        if time.monotonic() - self._last_valid_at > self.stale_after_s:
            return None
        return median(self._readings)

    def close(self) -> None:
        """Release ONLY this sensor's pins. Never GPIO.cleanup() globally —
        that would tear down the motor driver's pins mid-run."""
        if _ON_PI:
            GPIO.cleanup([self.pins.trig, self.pins.echo])

    # ── Hardware ──────────────────────────────────────────────────────────

    def _ping_cm(self) -> Optional[float]:
        """One trigger/echo cycle. Returns cm, or None on timeout/out-of-range."""
        if not _ON_PI:
            return None

        # 10 microsecond trigger pulse per the HC-SR04 datasheet.
        GPIO.output(self.pins.trig, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.pins.trig, GPIO.LOW)

        # Wait for echo to go high (sound burst sent).
        deadline = time.perf_counter() + _ECHO_RISE_TIMEOUT_S
        while GPIO.input(self.pins.echo) == 0:
            if time.perf_counter() > deadline:
                return None
        pulse_start = time.perf_counter()

        # Wait for echo to drop (reflection received).
        deadline = pulse_start + _ECHO_FALL_TIMEOUT_S
        while GPIO.input(self.pins.echo) == 1:
            if time.perf_counter() > deadline:
                return None
        pulse_end = time.perf_counter()

        cm = (pulse_end - pulse_start) * _CM_PER_SECOND_HALVED
        if cm < self.min_range_cm or cm > self.max_range_cm:
            return None
        return cm
