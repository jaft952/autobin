"""One background thread pings all ultrasonics in turn so control-loop reads never block.

The priority sensor (the front one) takes every other slot: [F, B, F, L, F, R].
Plain round-robin refreshed the front reading only every ~0.26 s, and with
the 3-sample median a new obstacle took ~0.8 s to register. Interleaving
doubles the front rate WITHOUT shortening the gap between any two pings, so
the acoustic spacing the HC-SR04 needs is untouched.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor

PING_INTERVAL_S = 0.06   # matches the HC-SR04 ~60ms between-measurement minimum


def build_schedule(sensors: List[UltrasonicSensor],
                   priority: Optional[UltrasonicSensor]) -> List[UltrasonicSensor]:
    if priority is None or priority not in sensors:
        return list(sensors)
    others = [s for s in sensors if s is not priority]
    if not others:
        return [priority]
    schedule: List[UltrasonicSensor] = []
    for sensor in others:
        schedule += [priority, sensor]
    return schedule


class UltrasonicArray:
    def __init__(self, sensors: List[Optional[UltrasonicSensor]],
                 ping_interval_s: float = PING_INTERVAL_S,
                 priority: Optional[UltrasonicSensor] = None):
        self._sensors = [s for s in sensors if s is not None]
        self._schedule = build_schedule(self._sensors, priority)
        self._ping_interval_s = ping_interval_s
        self._alive = False
        self._worker: Optional[threading.Thread] = None

    @property
    def schedule(self) -> List[UltrasonicSensor]:
        return list(self._schedule)

    def start(self) -> None:
        if not self._sensors or self._worker is not None:
            return
        self._alive = True
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="ultrasonic-worker")
        self._worker.start()

    def stop(self) -> None:
        self._alive = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        for sensor in self._sensors:
            sensor.close()

    def _run(self) -> None:
        index = 0
        while self._alive:
            self._schedule[index].update()
            index = (index + 1) % len(self._schedule)
            time.sleep(self._ping_interval_s)
