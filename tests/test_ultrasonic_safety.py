import pytest

from src.visual_servoing.ultrasonic_safety import UltrasonicSafety


class FakeUltrasonicSensor:
    def __init__(self, distance_cm):
        self._distance_cm = distance_cm

    def update(self):
        return None

    def get_distance_cm(self):
        return self._distance_cm

    def close(self):
        return None


def test_emergency_stop_is_unconditional_even_when_vision_agrees():
    # The ultrasonic must always be allowed to stop the robot first,
    # regardless of what vision says -- vision only gets consulted AFTER
    # the stop (by the caller, via matches_vision) to decide what happens
    # next. So even a perfect vision agreement must not suppress the stop
    # itself.
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(20.0) # type: ignore

    state = safety.update(vision_distance_cm=19.4)

    assert state.emergency_stop is True
    assert state.matches_vision is True
    assert state.grab_confirmed is True


def test_matches_vision_false_when_vision_disagrees():
    # Ultrasonic says 8cm but vision still says the can is far away -> the
    # ultrasonic is bouncing off something the camera isn't tracking (wall,
    # leg, etc.) -- matches_vision must be False so the caller scans instead
    # of driving toward it.
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(8.0) # type: ignore

    state = safety.update(vision_distance_cm=70.0)

    assert state.emergency_stop is True
    assert state.matches_vision is False


def test_matches_vision_false_when_no_vision_reading():
    # No target found by the camera this frame -> nothing to corroborate
    # the ultrasonic reading against, so it can't be confirmed as the can.
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(20.0) # type: ignore

    state = safety.update(vision_distance_cm=None)

    assert state.emergency_stop is True
    assert state.matches_vision is False


def test_can_grab_is_blocked_without_vision_confirmation():
    # grab_confirmed and emergency_stop share the same 25cm threshold, so
    # they're always true together at grab range -- can_grab must gate on
    # matches_vision instead, not "not emergency_stop" (which would always
    # be False here and permanently block grabbing).
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(20.0) # type: ignore

    state = safety.update()

    assert state.emergency_stop is True
    assert state.grab_confirmed is True
    assert safety.can_grab() is False


def test_can_grab_true_when_vision_confirms_target():
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(20.0) # type: ignore

    assert safety.can_grab(vision_distance_cm=19.4) is True
