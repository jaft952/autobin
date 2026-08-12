"""
High-level safety logic for the front ultrasonic sensors (top + bottom).

Responsibilities
----------------
1. Poll both HC-SR04s.
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

EMERGENCY_STOP_CM = 30.0
GRAB_CONFIRM_CM = 35.0


def _within(d: Optional[float], threshold_cm: float) -> bool:
    return d is not None and d <= threshold_cm


@dataclass
class UltrasonicState:
    distance_top_cm: Optional[float]
    distance_bottom_cm: Optional[float]
    emergency_stop: bool
    grab_confirmed: bool


class UltrasonicSafety:

    def __init__(
        self,
        trig: int = 23,
        echo: int = 24,
        trig2: int = 27,
        echo2: int = 22,
    ):
        self.sensor_top = UltrasonicSensor(
            UltrasonicPins(trig=trig, echo=echo)
        )
        self.sensor_bottom = UltrasonicSensor(
            UltrasonicPins(trig=trig2, echo=echo2)
        )

    def update(self) -> UltrasonicState:
        """
        Poll both sensors once and return the current safety state.

        emergency_stop is top OR bottom -- a close reading on EITHER sensor
        is reason enough to cut the wheels, regardless of what's calibrated
        where. grab_confirmed stays TOP-ONLY: it's gated against the arm
        solver's known ~29cm grasp distance, which only the top sensor's
        mounting was calibrated against (see GRAB_CONFIRM_CM above); ORing
        the bottom sensor in here would fire the arm on a close bottom
        reading (floor, wheel well, ground clutter) that has nothing to do
        with the tin actually being in grab position.
        """

        self.sensor_top.update()
        # Brief settling gap before the second sensor pings -- firing it
        # immediately after the top sensor's echo returns risks the top
        # transducer still ringing down / a stray reflection crossing over,
        # which reads as a bogus near-zero distance (see MIN_VALID_DISTANCE_CM
        # in ultrasonic_sensor.py, which now also guards against exactly that
        # as a second line of defense).
        time.sleep(0.01)
        self.sensor_bottom.update()

        d_top = self.sensor_top.get_distance_cm()
        d_bottom = self.sensor_bottom.get_distance_cm()

        return UltrasonicState(
            distance_top_cm=d_top,
            distance_bottom_cm=d_bottom,
            emergency_stop=(_within(d_top, EMERGENCY_STOP_CM) or
                             _within(d_bottom, EMERGENCY_STOP_CM)),
            grab_confirmed=_within(d_top, GRAB_CONFIRM_CM),
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
        self.sensor_top.close()
        self.sensor_bottom.close()


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
        self._latest = UltrasonicState(distance_top_cm=None, distance_bottom_cm=None,
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
                state = UltrasonicState(distance_top_cm=None, distance_bottom_cm=None,
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