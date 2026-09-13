"""Helpers for editing the arc grasp grid during calibration."""
from __future__ import annotations

from src.arm.arc_grasp import row_ny_at

POSE_ATTACH_CHANNELS = (1, 2, 3)
POSE_ATTACH_AMBIGUOUS_DEG = 15.0
POSE_ATTACH_NEW_ARC_DEG = 45.0


def choose_row_for_sample(rows, cur_arm, nx, ny):
    """Pick which grid row a new calibration sample belongs to."""
    if not rows:
        return None, float("inf"), "no rows"

    def dist(r):
        ss = r.get("samples") or []
        if not ss:
            return float("inf")
        return min(sum(abs(float(s["arm"][i]) - float(cur_arm[i]))
                       for i in POSE_ATTACH_CHANNELS) for s in ss)

    scored = sorted(((dist(r), r) for r in rows), key=lambda t: t[0])
    best_d, best = scored[0]
    note = ""
    if len(scored) > 1 and scored[1][0] - best_d < POSE_ATTACH_AMBIGUOUS_DEG:
        cands = [(d, r) for d, r in scored if d - best_d < POSE_ATTACH_AMBIGUOUS_DEG]
        best_d, best = min(cands, key=lambda t: abs(row_ny_at(t[1], nx) - ny))
        note = "postures ambiguous -> nearest curve decided"
    elif best_d > POSE_ATTACH_NEW_ARC_DEG:
        note = ("current CH2-4 matches NO row well — new distance? "
                "consider 'y' instead")
    return best, best_d, note
