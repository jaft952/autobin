"""
tests/calibrate_drive_speed.py

One-time speed calibration for the cruise-mode metric approach planner
(src/visual_servoing/approach_planner.py). Measures the real m/s and rad/s
this robot achieves at duty 1.0 and saves them to
src/visual_servoing/config/drive_model.yaml, so ApproachPlanner starts
correct instead of guessing.

WHY THIS EXISTS
---------------
ApproachPlanner's dead reckoning converts commanded duty to real distance via
DriveModel.v_max/w_max, and self-corrects online from vision as it drives.
That correction is bounded (it can't safely assume a 3-4x speed error away in
one approach), so a robot whose real top speed is far from the built-in
default (0.50 m/s, 3.0 rad/s) can overshoot before the online fit catches up
— exactly the "moves way too fast, flies off" failure this script exists to
prevent. Measuring it once, deliberately, with a static target and a known
test duty, removes the guess entirely: cruise starts already calibrated.

This mirrors how everything else on this robot gets calibrated (arc_grasp,
pixel_to_arm): put a real measurement in a file, don't infer it blind.

HOW IT WORKS
------------
Place an upright tin ~0.6-0.9 m ahead, centered in the camera. The script:
  1. measures the tin's floor position (via GroundRange, the same row->range
     fit cruise uses) BEFORE a short, known test move,
  2. commands a fixed duty for a fixed duration (forward, then turn),
  3. measures the tin's floor position AFTER,
  4. inverts the exact rigid-motion relationship (the same closed-form
     ApproachPlanner._decompose_motion uses) to recover how far it actually
     travelled / turned,
  5. divides by (test duty * duration) to get real m/s and rad/s,
  6. repeats a few times, averages, saves.

SAFETY: each move is short (well under a second) and at a moderate duty, but
the robot WILL drive briefly under open-loop command with no obstacle
avoidance. Clear >= 1.5 m directly ahead and to both sides before running.
You'll be asked to confirm before each of the forward/turn trials.

Usage: python tests/calibrate_drive_speed.py [--fwd-duty 0.3] [--turn-duty 0.3]
                                             [--trials 3]
"""
import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visual_servoing.approach_planner import (
    ApproachPlanner, DriveModel, GroundRange, DEFAULT_V_MAX, DEFAULT_W_MAX,
    V_MAX_BOUNDS, W_MAX_BOUNDS,
)

FWD_TEST_S = 0.7        # short: keeps displacement small enough to stay in frame
TURN_TEST_S = 0.6
SETTLE_S = 0.35         # let the base fully stop + camera catch up before re-measuring
MIN_RANGE_M = 0.15      # refuse to measure/trust a detection this close (noisy)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _make_detector():
    from src.perception.detector import AluminiumCanDetector, RUNTIME_MODEL_PATH
    det = AluminiumCanDetector(device="cpu", imgsz=640, model_path=RUNTIME_MODEL_PATH,
                               frame_width=1280, frame_height=720)
    det.start()
    return det


def _make_chassis():
    from src.visual_servoing.chassis_controller import ChassisController
    return ChassisController()


def _measure_floor_xy(det, ground: GroundRange, tries: int = 20):
    """Detect the tin and project its ground-contact point to (right, forward)
    metres via GroundRange. None if nothing usable is seen."""
    for _ in range(tries):
        frame = det.read_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        result = det.infer(frame)
        best = result.best
        if best is None:
            time.sleep(0.05)
            continue
        ori = getattr(best, "orientation", None)
        if ori is not None and ori.klass in ("lying", "axial"):
            print("  ! tin looks LYING — stand it upright for this calibration.")
            time.sleep(0.05)
            continue
        u, v = best.base_center
        p = ground.project(u, v, result.frame_width)
        if p is None:
            print("  ! detected, but out of the trusted ground-range band "
                  "(too far / above the horizon) — move it closer.")
            time.sleep(0.05)
            continue
        if p[1] < MIN_RANGE_M:
            print(f"  ! too close ({p[1]:.2f} m) to measure reliably — back it off.")
            time.sleep(0.05)
            continue
        return p
    return None


def _confirm(msg: str) -> bool:
    try:
        return input(f"{msg} [y/N]: ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def _trial_forward(det, ground, chassis, duty, duration):
    print(f"\n[forward trial] measuring BEFORE...")
    before = _measure_floor_xy(det, ground)
    if before is None:
        print("  no usable detection — skipping this trial.")
        return None
    print(f"  tin at (right={before[0]:+.3f}, forward={before[1]:.3f}) m")
    if not _confirm(f"  drive FORWARD at duty {duty} for {duration:.1f}s now?"):
        print("  skipped.")
        return None
    chassis.set_motor_pwm(duty, 0.0)
    time.sleep(duration)
    chassis.stop()
    time.sleep(SETTLE_S)
    after = _measure_floor_xy(det, ground)
    if after is None:
        print("  lost the tin after the move — discarding this trial.")
        return None
    print(f"  tin now at (right={after[0]:+.3f}, forward={after[1]:.3f}) m")

    # Tie-break the ambiguous root toward "drove roughly forward, barely
    # turned" — we commanded pure forward duty, so that prior is exactly
    # right, not just a rough guess.
    pred_s = duty * DEFAULT_V_MAX * duration
    dec = ApproachPlanner._decompose_motion(before, after, pred_s=pred_s, pred_psi=0.0)
    if dec is None:
        print("  measurement too close/noisy to invert — discarding.")
        return None
    s, psi = dec
    v_max = s / (duty * duration)
    print(f"  moved {s * 100:.1f} cm (turned {psi:+.3f} rad along the way) "
          f"-> v_max sample {v_max:.3f} m/s")
    return v_max


def _trial_turn(det, ground, chassis, duty, duration):
    print(f"\n[turn trial] measuring BEFORE...")
    before = _measure_floor_xy(det, ground)
    if before is None:
        print("  no usable detection — skipping this trial.")
        return None
    print(f"  tin at (right={before[0]:+.3f}, forward={before[1]:.3f}) m")
    if not _confirm(f"  TURN (steer duty {duty}) for {duration:.1f}s now?"):
        print("  skipped.")
        return None
    chassis.set_motor_pwm(0.0, duty)
    time.sleep(duration)
    chassis.stop()
    time.sleep(SETTLE_S)
    after = _measure_floor_xy(det, ground)
    if after is None:
        print("  lost the tin after the move — discarding this trial.")
        return None
    print(f"  tin now at (right={after[0]:+.3f}, forward={after[1]:.3f}) m")

    pred_psi = duty * DEFAULT_W_MAX * duration
    dec = ApproachPlanner._decompose_motion(before, after, pred_s=0.0, pred_psi=pred_psi)
    if dec is None:
        print("  measurement too close/noisy to invert — discarding.")
        return None
    s, psi = dec
    w_max = abs(psi) / (duration * abs(duty))
    print(f"  turned {math.degrees(psi):+.1f} deg (drifted {s * 100:.1f} cm along "
          f"the way) -> w_max sample {w_max:.3f} rad/s")
    return w_max


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Measure real drive speed for ApproachPlanner")
    ap.add_argument("--fwd-duty", type=float, default=0.3)
    ap.add_argument("--turn-duty", type=float, default=0.3)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    ground = GroundRange()
    print(f"[cfg] {ground.status()}")
    if not ground.ready:
        print("No ground-range calibration (no ruler samples in "
              "centering_config.yaml / pixel_to_arm.yaml) — this tool has "
              "nothing to measure distance with. Calibrate that first.")
        return

    print("\n" + "=" * 70)
    print("DRIVE SPEED CALIBRATION")
    print("Place an UPRIGHT tin ~0.6-0.9 m directly ahead, centered in view.")
    print("Clear >= 1.5 m ahead and to both sides — the base WILL move.")
    print("=" * 70)
    if not _confirm("\nReady to start"):
        print("cancelled.")
        return

    try:
        det = _make_detector()
    except Exception as exc:
        print(f"camera not available ({exc}) — can't calibrate without it.")
        return
    try:
        chassis = _make_chassis()
    except Exception as exc:
        print(f"chassis not available ({exc}) — can't calibrate without it.")
        det.stop()
        return

    v_samples, w_samples = [], []
    try:
        for i in range(args.trials):
            print(f"\n--- forward trial {i + 1}/{args.trials} ---")
            v = _trial_forward(det, ground, chassis, args.fwd_duty, FWD_TEST_S)
            if v is not None and V_MAX_BOUNDS[0] <= v <= V_MAX_BOUNDS[1] * 1.5:
                v_samples.append(v)
        for i in range(args.trials):
            print(f"\n--- turn trial {i + 1}/{args.trials} ---")
            w = _trial_turn(det, ground, chassis, args.turn_duty, TURN_TEST_S)
            if w is not None and W_MAX_BOUNDS[0] <= w <= W_MAX_BOUNDS[1] * 1.5:
                w_samples.append(w)
    finally:
        chassis.stop()
        chassis.close()
        det.stop()

    print("\n" + "=" * 70)
    if not v_samples and not w_samples:
        print("No usable measurements — nothing saved. Check tin placement / "
              "clearance and try again.")
        return

    model = DriveModel()
    if v_samples:
        avg_v = sum(v_samples) / len(v_samples)
        spread = max(v_samples) - min(v_samples) if len(v_samples) > 1 else 0.0
        print(f"v_max: {len(v_samples)} samples, mean {avg_v:.3f} m/s "
              f"(spread {spread:.3f})")
        model.v_max = _clamp(avg_v, *V_MAX_BOUNDS)
        model._dirty = True
    else:
        print("v_max: no usable forward samples — keeping previous value "
              f"({model.v_max:.3f} m/s).")
    if w_samples:
        avg_w = sum(w_samples) / len(w_samples)
        spread = max(w_samples) - min(w_samples) if len(w_samples) > 1 else 0.0
        print(f"w_max: {len(w_samples)} samples, mean {avg_w:.3f} rad/s "
              f"(spread {spread:.3f})")
        model.w_max = _clamp(avg_w, *W_MAX_BOUNDS)
        model._dirty = True
    else:
        print("w_max: no usable turn samples — keeping previous value "
              f"({model.w_max:.3f} rad/s).")

    model.updates = max(model.updates, 6)   # start ApproachPlanner at full trust
    model._dirty = True
    model.save()
    print(f"\nsaved -> {model.path}")
    print(model.status())
    print("\ncruise mode will load this on next launch of test_ibvs_centering.py.")


if __name__ == "__main__":
    main()
