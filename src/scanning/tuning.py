"""All tunable constants for the zigzag scan layer. Read qualified
(tuning.X) elsewhere so tests/runtime can monkeypatch a single value."""

FORWARD_SPEED = 0.20   # lane cruising speed
TURN_SPEED    = 0.30   # pivot speed; below this the base tends to stall

TURN_AT_CM = 30.0   # front sensor: end the lane and start the zigzag turn
                     # (kept above SensorHub.EMERGENCY_STOP_CM by design: Layer 1
                     # should turn away from a wall on its own well before Layer 5's
                     # hard stop is needed as a backstop, not after it)

DIAGONAL_NUDGE_CM = 30.0   # side distance that starts the steering nudge
NUDGE_GAIN        = 0.35   # max nudge strength, fraction of turn speed

PIVOT_ROOM_CM   = 40.0   # side-room reading cap used when comparing left/right
PIVOT_TIGHT_CM  = 25.0   # side distance considered too tight to pivot into
PIVOT_MARGIN_CM = 5.0    # other side must be this much roomier to override alternation

TURN_90_S  = 0.9   # calibrated seconds for a ~90 deg pivot
SHIFT_S    = 1.2   # calibrated seconds to drive ~one lane width
MAX_LANE_S = 30.0  # longest straight run before turning anyway
SETTLE_S   = 0.15  # pause between phases so wheels stop before reversing

TIMING_JITTER = 0.2   # +/- fraction randomizing every timed phase
BOUNCE_EXTRA  = 1.0    # TURN2 sweeps 90..90*(1+this) deg instead of a fixed 90
