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


def test_can_grab_is_blocked_when_emergency_stop_is_active():
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(20.0) # type: ignore

    state = safety.update()

    assert state.emergency_stop is True
    assert state.grab_confirmed is True
    assert safety.can_grab() is False


def test_emergency_stop_suppressed_when_vision_agrees():
    # Ultrasonic and vision both say ~20cm -> same object (the tracked can),
    # so the raw emergency stop should stand down and let
    # reactive_controller's own too_close/reached handling take over.
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(20.0) # type: ignore

    state = safety.update(vision_distance_cm=19.4)

    assert state.matches_vision is True
    assert state.emergency_stop is False
    # grab confirmation is unaffected by the agreement check
    assert state.grab_confirmed is True


def test_emergency_stop_stays_active_when_vision_disagrees():
    # Ultrasonic says 8cm but vision still says the can is far away -> the
    # ultrasonic is bouncing off something the camera isn't tracking (wall,
    # leg, etc.), so the hard stop must stay in force.
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(8.0) # type: ignore

    state = safety.update(vision_distance_cm=70.0)

    assert state.matches_vision is False
    assert state.emergency_stop is True


def test_emergency_stop_stays_active_when_no_vision_reading():
    # No target found by the camera this frame -> nothing to corroborate
    # the ultrasonic reading against, so it stays a hard stop.
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(20.0) # type: ignore

    state = safety.update(vision_distance_cm=None)

    assert state.matches_vision is False
    assert state.emergency_stop is True
