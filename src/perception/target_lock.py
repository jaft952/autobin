"""
src/perception/target_lock.py

TargetLock — picks ONE tin to work on and sticks to it.

Policy:
  1. No target held -> lock the detection with the LARGEST bbox area
     (largest = nearest/most complete, so it is the cheapest to collect).
  2. Target held -> every later frame is matched back to that same tin and
     ALL other detections are ignored, so the base never re-aims mid-approach
     when a second tin drifts into view.
  3. Target gone from view for LOST_GRACE_S -> release the lock and pick the
     new largest. A successful grab removes the tin from the frame, so this
     is what advances the robot to the next tin.

The grace window matters more than it looks: the arm occludes the camera
during a grab and the tick loop is blocked for seconds, so a "missing for
one frame" rule would drop the lock on a tin that is still there.

Matching is greedy nearest-center, gated by distance and area ratio so a
different tin passing nearby cannot steal the lock.

Pure math on plain data classes — no camera, no torch, no hardware imports.
"""
from __future__ import annotations

import math
import time
from typing import Any, List, Optional

# Release the lock when the tin has not been matched for this long.
LOST_GRACE_S = 1.5

# A candidate matches the held target only if its center moved less than
# MATCH_DIST_FACTOR * (target box size) and its area is within the ratio
# window. Wide on purpose: the box grows fast while the base drives in.
MATCH_DIST_FACTOR = 0.9
MATCH_DIST_MIN_PX = 60.0
MATCH_AREA_MIN_RATIO = 0.25
MATCH_AREA_MAX_RATIO = 4.0


def _area(box: Any) -> float:
    return float(max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1))


class TargetLock:
    """Holds the current tin of interest across frames."""

    def __init__(self, lost_grace_s: float = LOST_GRACE_S):
        self._lost_grace_s = lost_grace_s
        self._target: Optional[Any] = None
        self._last_seen: float = 0.0

    @property
    def locked(self) -> bool:
        return self._target is not None

    @property
    def target(self) -> Optional[Any]:
        """Last matched box of the held tin (None when nothing is locked)."""
        return self._target

    def release(self) -> None:
        """Drop the lock now — next select() picks the largest again."""
        self._target = None
        self._last_seen = 0.0

    def select(self, detections: List[Any], now: Optional[float] = None) -> Optional[Any]:
        """Return the box to act on this frame, or None.

        None while the held tin is missing but still inside the grace window:
        the caller must NOT fall back to another tin there.
        """
        now = time.monotonic() if now is None else now

        if self._target is not None:
            match = self._match(detections)
            if match is not None:
                self._target = match
                self._last_seen = now
                return match
            if now - self._last_seen < self._lost_grace_s:
                return None                  # occluded/blinking — keep waiting
            self.release()

        if not detections:
            return None
        self._target = max(detections, key=_area)
        self._last_seen = now
        return self._target

    # ── Internals ─────────────────────────────────────────────────────────

    def _match(self, detections: List[Any]) -> Optional[Any]:
        t = self._target
        if t is None or not detections:
            return None
        t_area = _area(t)
        max_dist = max(MATCH_DIST_MIN_PX,
                       MATCH_DIST_FACTOR * max(t.x2 - t.x1, t.y2 - t.y1))
        best, best_dist = None, float("inf")
        for d in detections:
            dist = math.hypot(d.center_x - t.center_x, d.center_y - t.center_y)
            if dist > max_dist or dist >= best_dist:
                continue
            if t_area > 0:
                ratio = _area(d) / t_area
                if not (MATCH_AREA_MIN_RATIO <= ratio <= MATCH_AREA_MAX_RATIO):
                    continue
            best, best_dist = d, dist
        return best
