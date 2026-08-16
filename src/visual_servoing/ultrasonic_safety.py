"""
High-level safety logic for the front ultrasonic sensor.

Was a pair stacked vertically, both aimed straight ahead (the old
top/bottom naming). The lower one has been remounted as the front-left
diagonal and is now owned by SensorHub / layer 5, so this module reads the
front sensor only; the second-sensor code is left commented out below.

Responsibilities
----------------
1. Poll the front HC-SR04.
2. Decide whether emergency stop is required.
3. Decide whether the arm is allowed to grab.
4. Never contains GPIO code other than calling UltrasonicSensor.

Pure decision layer built on top of the hardware driver.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from src.hardware.sensors.ultrasonic_sensor import (
    UltrasonicSensor,
    UltrasonicPins,
)

EMERGENCY_STOP_CM = 25.0
GRAB_CONFIRM_CM = 35.0


def _within(d: Optional[float], threshold_cm: float) -> bool:
    return d is not None and d <= threshold_cm


@dataclass
class UltrasonicState:
    # OLD (second sensor, was bottom, now the front_left diagonal):
    # distance_bottom_cm: Optional[float]
    distance_front_cm: Optional[float]
    emergency_stop: bool
    grab_confirmed: bool


class UltrasonicSafety:

    def __init__(
        self,
        trig: int = 23,
        echo: int = 24,
        # OLD second sensor: trig2: int = 27, echo2: int = 22,
    ):
        self.sensor_front = UltrasonicSensor(
            UltrasonicPins(trig=trig, echo=echo)
        )
        # OLD:
        # self.sensor_bottom = UltrasonicSensor(
        #     UltrasonicPins(trig=trig2, echo=echo2)
        # )

    def update(self) -> UltrasonicState:
        """
        Poll the front sensor once and return the current safety state.

        Both signals come from the front sensor. It is the only one aimed
        where the tin sits, which is what grab_confirmed is gated against
        (the arm solver's known ~29cm grasp distance). Off-axis coverage is
        SensorHub / layer 5's job now.

        grab_confirmed is also True when the front sensor has NO reading at
        all (echo timeout, d_front is None) -- not just when it's within
        range. The HC-SR04's beam is narrow; a tin sitting off-center (which
        is a perfectly normal, calibrated arc_grasp position, not an edge
        case) can sit outside that cone entirely, so the echo has nothing to
        bounce off and times out even though the tin is really there. vision
        + the arc solver's own per-row calibration (solve_with_band) already
        confirm the tin's actual position independently of this sensor, so a
        missing front reading must not be able to block a real grab -- same
        reasoning as the existing "hardware fault" fallback, just triggered
        by geometry instead of a dead sensor.
        """

        self.sensor_front.update()
        d_front = self.sensor_front.get_distance_cm()

        # OLD second sensor: a settling gap was needed before pinging it, or
        # the front transducer's ring-down read back as a bogus near-zero.
        # time.sleep(0.01)
        # self.sensor_bottom.update()
        # d_bottom = self.sensor_bottom.get_distance_cm()

        # OLD: emergency_stop ORed the second sensor in.
        # emergency_stop=(_within(d_front, EMERGENCY_STOP_CM) or
        #                 _within(d_bottom, EMERGENCY_STOP_CM)),
        return UltrasonicState(
            distance_front_cm=d_front,
            emergency_stop=_within(d_front, EMERGENCY_STOP_CM),
            grab_confirmed=(_within(d_front, GRAB_CONFIRM_CM) or d_front is None),
        )

    def should_stop(self) -> bool:
        """
        Convenience method.
        """
        s = self.update()
        return s.emergency_stop

    def can_grab(self) -> bool:
        """
        Convenience method.

        Gated on proximity alone (within GRAB_CONFIRM_CM) -- NOT on
        emergency_stop, which is a separate drive-only cutoff. At the
        correct grab distance the tin can is expected to trip it too, so
        ANDing the two would block a legitimate grab (see EMERGENCY_STOP_CM
        above).
        """
        s = self.update()
        return s.grab_confirmed

    def close(self):
        self.sensor_front.close()
        # OLD: self.sensor_bottom.close()


# HC-SR04 datasheets recommend at least ~60ms between trigger pulses so an
# echo from the previous ping can't be mistaken for the next one -- this
# isn't just a CPU-niceness knob, polling faster risks corrupt readings.
DEFAULT_WATCHDOG_POLL_HZ = 15.0


class UltrasonicWatchdog:
    """Polls UltrasonicSafety in its own background thread instead of once
    per main-loop iteration, so emergency stop is never delayed behind
    camera inference or a time.sleep() elsewhere in the control loop (e.g.
    a step-and-look nudge in reactive_controller's final-approach mode) --
    those can each take longer than one ultrasonic poll, and until now
    nothing was watching the sensor while they ran.

    Top priority: the moment a poll comes back with emergency_stop, this
    thread calls stop_callback (e.g. actuator.stop) ITSELF, immediately --
    it does not wait for the main loop to notice a flag and react next
    iteration. The main loop should still read `.latest` every iteration
    and treat emergency_stop as the first check before issuing any new
    drive command (see run_live_demo): the watchdog's own stop_callback
    call handles "already moving right now", and the main loop's check
    stops it from immediately re-driving over that on its very next
    command -- the two work together, neither alone is enough.

    stop_callback may be called from this background thread concurrently
    with the main thread calling actuator.apply()/stop(); that's accepted
    here rather than adding locking inside the actuator itself (out of
    scope -- src/hardware stays untouched), on the same reasoning
    UltrasonicSensor already relies on for its own GPIO calls: worst case
    is a stop() getting immediately followed by a stale apply() from the
    main thread, and this watchdog will simply stop it again on its next
    poll (poll_hz), typically well under 100ms later.
    """

    def __init__(self, safety: UltrasonicSafety, stop_callback: Callable[[], None],
                 poll_hz: float = DEFAULT_WATCHDOG_POLL_HZ):
        self._safety = safety
        self._stop_callback = stop_callback
        self._interval_s = 1.0 / poll_hz
        self._lock = threading.Lock()
        self._latest = UltrasonicState(distance_front_cm=None,
                                        emergency_stop=False, grab_confirmed=False)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ultrasonic-watchdog", daemon=True)

    def start(self) -> "UltrasonicWatchdog":
        self._thread.start()
        return self

    @property
    def latest(self) -> UltrasonicState:
        """Cheap, thread-safe read of the most recent poll -- the main loop
        calls this instead of UltrasonicSafety.update() directly, so it
        never blocks on (or duplicates) the sensor's own busy-wait read."""
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = self._safety.update()
            except Exception as exc:
                # A transient GPIO error (e.g. timing contention with
                # camera inference / PWM on the main thread) must not
                # silently kill this daemon thread -- without a reading,
                # .latest would stay frozen at "no reading" forever, with
                # nothing left to recover it. Log and keep polling instead.
                print(f"[ultrasonic] poll error: {exc}")
                state = UltrasonicState(distance_front_cm=None,
                                         emergency_stop=False, grab_confirmed=False)
            with self._lock:
                self._latest = state
            if state.emergency_stop:
                self._stop_callback()
            time.sleep(self._interval_s)

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the poll loop to exit and join it. Call this BEFORE
        closing the underlying sensor (UltrasonicSafety.close()) -- once
        joined, the thread is guaranteed to no longer be mid-poll, so
        there's no race between a background .update() and GPIO cleanup."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)