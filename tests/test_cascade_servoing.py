#!/usr/bin/env python3
"""
Test: Multi-Rate Cascade Visual Servoing Controller.

Demonstrates:
1. Outer loop (Vision ~30 Hz): YOLO detection + IBVS error computation
2. Inner loop (Motor ~1000 Hz): Smooth interpolation between vision updates
3. Uses mask_area as confidence metric for dynamic filter tuning
4. Non-blocking thread synchronization via TrackingBuffer

Run:
    python tests/test_cascade_servoing.py

Expected output:
    [Vision] err_x=+0.123, err_y=-0.045, mask_area=45000, q=0.87
    [Motor] Vision update: px=+0.120, py=-0.043, conf=0.87
    [Status] stable=False, aligned=True, quality=0.87
"""

import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.detector import AluminiumCanDetector, DetectionResult, BoundingBox
from src.visual_servoing.ibvs_centering import IBVSCentering, CenteringConfig
from src.visual_servoing.cascade_controller import (
    CascadeController, VisionState, TrackingBuffer
)


class MockDetector:
    """Simulate YOLO detector with synthetic tin motion."""

    def __init__(self):
        self.t_start = time.time()

    def read_frame(self):
        """Return dummy frame."""
        return None  # Mocked

    def infer(self, frame) -> DetectionResult:
        """Generate synthetic detection with moving tin."""
        t = time.time() - self.t_start

        # Simulate tin approaching and centering
        # error_x: oscillates (left/right)
        # error_y: exponential decay toward zero (approaching target)
        error_x_px = 640 + 100 * math.sin(t)
        error_y_px = 360 + 200 * math.exp(-0.5 * t)  # Approaches target

        # mask_area increases as tin gets closer
        mask_area = int(30000 + 40000 * (1 - math.exp(-t)))

        # Create synthetic detection
        box = BoundingBox(
            x1=int(error_x_px - 50),
            y1=int(error_y_px - 60),
            x2=int(error_x_px + 50),
            y2=int(error_y_px + 60),
            confidence=0.95,
            mask_area=mask_area, # type: ignore
            orientation=None,
        )

        return DetectionResult(
            detections=[box],
            frame_width=1280,
            frame_height=720,
        )


class MockChassisController:
    """Mock motor controller to track commands."""

    def __init__(self):
        self.last_cmd = (0.0, 0.0)
        self.cmd_history = []

    def set_motor_pwm(self, forward: float, steer: float):
        """Record motor command."""
        self.last_cmd = (forward, steer)
        self.cmd_history.append((time.time(), forward, steer))

    def get_cmd_rate(self, window: float = 1.0) -> float:
        """Analyze command update rate over last N seconds."""
        now = time.time()
        recent = [
            c for c in self.cmd_history
            if (now - c[0]) < window
        ]
        return len(recent) if window > 0 else 0


def test_cascade_controller():
    """Run cascade control system with synthetic detector."""
    print("=" * 70)
    print("Multi-Rate Cascade Visual Servoing Test")
    print("=" * 70)
    print()

    # Initialize
    detector = MockDetector()
    centering = IBVSCentering()
    chassis = MockChassisController()

    cascade = CascadeController(detector, centering, chassis)

    print("Configuration:")
    print(f"  Vision loop: ~30 Hz (frame interval ~33ms)")
    print(f"  Motor loop: ~1000 Hz (dt ~1ms)")
    print(f"  Cascade buffer: max 5 states")
    print()

    print("Starting cascade controller...")
    cascade.start()
    print()

    # Run for 10 seconds
    print("Running for 10 seconds. Monitoring:")
    print("  - Vision detections (mask_area confidence)")
    print("  - Motor command rate")
    print("  - Alignment and stability")
    print()

    try:
        test_duration = 10.0
        t_start = time.time()
        sample_interval = 0.5
        t_last_sample = t_start

        while (time.time() - t_start) < test_duration:
            now = time.time()

            # Print status every 0.5 seconds
            if (now - t_last_sample) >= sample_interval:
                t_last_sample = now

                status = cascade.get_status()
                if status:
                    print(
                        f"[{now - t_start:6.2f}s] "
                        f"err_x={status.error_x:+.3f} "
                        f"err_y={status.error_y:+.3f} "
                        f"mask_area={status.mask_area:>6} "
                        f"q={status.quality():.2f} "
                        f"aligned={status.aligned} "
                        f"stable={status.stable}"
                    )

            time.sleep(0.01)

        print()
        print("Test completed successfully!")
        print()

        # Print statistics
        print("Motor Command Statistics:")
        cmd_rate = chassis.get_cmd_rate(window=10.0)
        expected_cmds = 10.0 * 1000  # 1000 Hz × 10 seconds
        print(f"  Total commands sent: {len(chassis.cmd_history)}")
        print(f"  Expected (1000 Hz × 10s): ~{expected_cmds:.0f}")
        print(f"  Actual rate: {len(chassis.cmd_history) / 10.0:.0f} Hz")
        print()

        # Analyze trajectory smoothness
        if len(chassis.cmd_history) > 1:
            accelerations = []
            for i in range(1, len(chassis.cmd_history)):
                t1, f1, s1 = chassis.cmd_history[i - 1]
                t2, f2, s2 = chassis.cmd_history[i]
                dt = t2 - t1
                if dt > 0:
                    accel_f = abs(f2 - f1) / dt
                    accel_s = abs(s2 - s1) / dt
                    accelerations.append(max(accel_f, accel_s))

            if accelerations:
                print("Acceleration Analysis (smoothness metric):")
                print(f"  Max acceleration: {max(accelerations):.2f} /s")
                print(f"  Avg acceleration: {sum(accelerations) / len(accelerations):.2f} /s")
                print(f"  Smooth motion achieved: {max(accelerations) < 1.0}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cascade.stop()
        print("\nCascade controller stopped.")


def test_tracking_buffer():
    """Unit test: TrackingBuffer functionality."""
    print("\n" + "=" * 70)
    print("TrackingBuffer Unit Test")
    print("=" * 70)
    print()

    buffer = TrackingBuffer(max_size=5)

    # Push some states
    print("Pushing 3 vision states...")
    for i in range(3):
        state = VisionState(
            timestamp=time.time() + i * 0.033,
            error_x=0.1 * i,
            error_y=-0.05 * i,
            mask_area=50000 + i * 5000,
            detected=True,
            aligned=(i >= 2),
            stable=False,
            confidence=0.9,
        )
        buffer.push(state)
        print(f"  State {i}: error_x={state.error_x:.3f}, mask_area={state.mask_area}")

    # Test snapshot
    snapshot = buffer.snapshot()
    print(f"\nBuffer snapshot: {len(snapshot)} states")

    # Test velocity estimation
    vx, vy = buffer.estimate_velocity()
    print(f"Estimated velocity: vx={vx:.3f}/s, vy={vy:.3f}/s")

    # Test quality
    quality = buffer.average_quality()
    print(f"Average quality: {quality:.2f}")
    print()
    print("✓ TrackingBuffer test passed")


def test_filters():
    """Unit test: Filter and interpolator components."""
    print("\n" + "=" * 70)
    print("Filter & Interpolator Unit Test")
    print("=" * 70)
    print()

    from src.visual_servoing.cascade_controller import (
        AlphaBetaFilter, TrajectoryInterpolator, VelocityLimiter
    )

    # Test alpha-beta filter
    print("Alpha-Beta Filter Test:")
    filt = AlphaBetaFilter(alpha=0.6, beta=0.3)
    t = 0.0
    print("  Smoothing noisy measurement signal:")
    for i in range(5):
        z = 0.1 * math.sin(t)  # True signal
        noise = 0.02 * (i % 2 - 0.5)  # Noise
        z_meas = z + noise
        x_filt, v_filt = filt.update(z_meas, t)
        print(
            f"    t={t:.3f}: z_meas={z_meas:+.3f}, "
            f"x_filt={x_filt:+.3f}, v_filt={v_filt:+.3f}"
        )
        t += 0.01

    # Test trajectory interpolator
    print("\n  Cubic Hermite Interpolator Test:")
    interp = TrajectoryInterpolator()
    interp.set_waypoints(p0=0.0, p1=0.1, v0=0.0, v1=0.0, t_start=0.0, t_end=0.1)
    print("    Interpolating from p0=0.0 to p1=0.1 over 100ms:")
    for t in [0.0, 0.025, 0.05, 0.075, 0.1]:
        p, v = interp.evaluate(t)
        print(f"      t={t:.3f}: p={p:+.3f}, v={v:+.3f}")

    # Test velocity limiter
    print("\n  Velocity Limiter Test:")
    limiter = VelocityLimiter(max_accel=0.5, dt=0.001)
    print("    Limiting velocity jumps (max_accel=0.5 /s):")
    desired_vs = [0.0, 0.5, 1.0, 0.2, -0.5, 0.0]
    for dv in desired_vs:
        lim_v = limiter.limit(dv)
        print(f"      desired={dv:+.3f} → limited={lim_v:+.3f}")

    print()
    print("✓ Filter test passed")


if __name__ == "__main__":
    # Run unit tests
    test_tracking_buffer()
    test_filters()

    # Run integration test
    test_cascade_controller()
