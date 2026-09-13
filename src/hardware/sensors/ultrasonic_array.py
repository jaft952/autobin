"""Reads all ultrasonic sensors in one background thread."""
from __future__ import annotations

import threading
import time
from typing import List, Optional

from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor

PING_INTERVAL_S = 0.06


class UltrasonicArray:
    def __init__(self, sensors: List[Optional[UltrasonicSensor]],
                 ping_interval_s: float = PING_INTERVAL_S):
        self._sensors = [s for s in sensors if s is not None]
        self._ping_interval_s = ping_interval_s
        self._alive = False
        self._worker: Optional[threading.Thread] = None

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
            self._sensors[index].update()
            index = (index + 1) % len(self._sensors)
            time.sleep(self._ping_interval_s)
