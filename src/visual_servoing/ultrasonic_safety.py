"""
High-level safety logic for the front ultrasonic sensor.

Responsibilities
----------------
1. Poll the HC-SR04.
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

# Two independent signals, not a nested pair -- emergency_stop is a DRIVE-only
# safety cutoff (stop/retreat the wheels so they don't ram into something);
# grab_confirmed is the ARM's proximity gate. They must stay independent: the
# tin can is EXPECTED to be this close at the correct grab position (measured:
# solver says grabbable at ~29cm), so treating "close" as universally
# dangerous would block a legitimate grab. See can_grab()/UltrasonicState.
EMERGENCY_STOP_CM = 30.0
GRAB_CONFIRM_CM = 35.0


@dataclass
class UltrasonicState:
    distance_cm: Optional[float]
    emergency_stop: bool
    grab_confirmed: bool


class UltrasonicSafety:

    def __init__(
        self,
        trig: int = 23,
        echo: int = 24,
    ):
        self.sensor = UltrasonicSensor(
            UltrasonicPins(trig=trig, echo=echo)
        )

    def update(self) -> UltrasonicState:
        """
        Poll the sensor once and return the current safety state.
        """

        self.sensor.update()

        d = self.sensor.get_distance_cm()

        if d is None:
            return UltrasonicState(
                distance_cm=None,
                emergency_stop=False,
                grab_confirmed=False,
            )

        return UltrasonicState(
            distance_cm=d,
            emergency_stop=d <= EMERGENCY_STOP_CM,
            grab_confirmed=d <= GRAB_CONFIRM_CM,
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
        self.sensor.close()


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
        self._latest = UltrasonicState(distance_cm=None, emergency_stop=False, grab_confirmed=False)
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
                state = UltrasonicState(distance_cm=None, emergency_stop=False,
                                         grab_confirmed=False)
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