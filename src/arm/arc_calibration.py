"""Grid-editing helpers for tests/test_arc_grasp.py ('a' = add sample).

Split out of arc_grasp.py (the runtime solver) since this is calibration-only
logic with exactly one caller.
"""
from __future__ import annotations

from src.arm.arc_grasp import row_ny_at

# Samples on ONE arc share the CH2/CH3/CH4 fold (one radius = one posture);
# only CH1 (azimuth) and CH5 (roll baseline) vary along it, so posture is the
# reliable attach key — nearest-curve-ny alone mis-attaches edge samples
# (the arc dips at the edges, but the curve doesn't know that yet).
POSE_ATTACH_CHANNELS = (1, 2, 3)      # CH2, CH3, CH4
POSE_ATTACH_AMBIGUOUS_DEG = 15.0      # postures closer than this: ny decides
POSE_ATTACH_NEW_ARC_DEG = 45.0        # nothing this close: probably a new arc


def choose_row_for_sample(rows, cur_arm, nx, ny):
    """Pick the row a new sample belongs to: best CH2-4 posture match,
    falling back to nearest curve-ny only when postures are ambiguous.
    Returns (row, posture_dist_deg, note)."""
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
