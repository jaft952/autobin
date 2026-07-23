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

EMERGENCY_STOP_CM = 25.0
GRAB_CONFIRM_CM = 25.0

# How closely the ultrasonic reading must agree with the camera's own
# monocular distance estimate (TargetError.distance_cm) to trust that both
# sensors are looking at the SAME object -- the tracked tin can, not a wall,
# table leg, or someone's foot that happens to be in the ultrasonic's cone
# but outside/unrecognized in the camera's view. Only ever consulted when
# the ultrasonic has already read <= EMERGENCY_STOP_CM, so this doesn't
# widen the near-field trigger zone -- it only decides, once something is
# that close, whether it's the can we're deliberately closing in on.
# TODO tune: wide enough to absorb sensor noise + a stale vision frame,
# tight enough that a real, unrelated obstacle can't accidentally "match".
VISION_AGREEMENT_TOLERANCE_CM = 8.0


@dataclass
class UltrasonicState:
    distance_cm: Optional[float]
    emergency_stop: bool
    grab_confirmed: bool
    matches_vision: bool = False


class UltrasonicSafety:

    def __init__(
        self,
        trig: int = 23,
        echo: int = 24,
    ):
        self.sensor = UltrasonicSensor(
            UltrasonicPins(trig=trig, echo=echo)
        )

    def update(self, vision_distance_cm: Optional[float] = None) -> UltrasonicState:
        """
        Poll the sensor once and return the current safety state.

        vision_distance_cm: the visual-servoing distance estimate for the
        currently tracked target (TargetError.distance_cm), if available.
        Pass None (the default) when there's no camera/target reading to
        compare against -- e.g. no target found this frame -- which keeps
        the old behavior of a hard stop on any close reading.

        When the ultrasonic already reads within EMERGENCY_STOP_CM AND that
        reading roughly agrees with vision_distance_cm (within
        VISION_AGREEMENT_TOLERANCE_CM), the ultrasonic is almost certainly
        bouncing off the same can the camera is tracking as it closes in for
        the grab, not an unrelated obstacle -- so emergency_stop is
        suppressed and reactive_controller's own too_close/reached handling
        (a gentle backward nudge, or the arm handoff) is trusted to do the
        right thing instead of a raw full stop. No vision reading, or a
        disagreement between the two, means the ultrasonic may be seeing
        something the camera isn't accounting for, so the hard stop stays.
        """

        self.sensor.update()

        d = self.sensor.get_distance_cm()

        if d is None:
            return UltrasonicState(
                distance_cm=None,
                emergency_stop=False,
                grab_confirmed=False,
                matches_vision=False,
            )

        matches_vision = (vision_distance_cm is not None
                           and abs(d - vision_distance_cm) <= VISION_AGREEMENT_TOLERANCE_CM)

        return UltrasonicState(
            distance_cm=d,
            emergency_stop=(d <= EMERGENCY_STOP_CM) and not matches_vision,
            grab_confirmed=d <= GRAB_CONFIRM_CM,
            matches_vision=matches_vision,
        )

    def should_stop(self, vision_distance_cm: Optional[float] = None) -> bool:
        """
        Convenience method.
        """
        s = self.update(vision_distance_cm=vision_distance_cm)
        return s.emergency_stop

    def can_grab(self, vision_distance_cm: Optional[float] = None) -> bool:
        """
        Convenience method.

        Grabbing is only allowed when the sensor is within the grab-confirm
        range and no emergency stop condition is active.
        """
        s = self.update(vision_distance_cm=vision_distance_cm)
        return s.grab_confirmed and not s.emergency_stop

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
        self._vision_distance_cm: Optional[float] = None
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

    def set_vision_distance(self, distance_cm: Optional[float]) -> None:
        """Hand the background poll loop the latest camera-based distance
        estimate (TargetError.distance_cm), so it can tell an in-range
        ultrasonic reading of the tracked can apart from an unrelated
        obstacle -- see UltrasonicSafety.update()'s vision_distance_cm.

        Call this every frame from the main loop right after
        compute_target_error() (pass None when the target isn't found this
        frame). The watchdog polls on its own cadence and simply reads
        whatever was set here most recently -- the same latest-value pattern
        `.latest` uses in the other direction -- so the value it compares
        against may be up to one camera frame stale. That's the same order
        of staleness the rest of this control loop already tolerates (see
        reactive_controller's step-and-look docstring) and is fine here:
        worst case is one extra poll before a suppressed stop re-engages, or
        vice versa.
        """
        with self._lock:
            self._vision_distance_cm = distance_cm

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                vision_distance_cm = self._vision_distance_cm
            state = self._safety.update(vision_distance_cm=vision_distance_cm)
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