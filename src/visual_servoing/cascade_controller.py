"""
Multi-Rate Cascade Visual Servoing Controller.

Architecture:
  Outer Loop (Vision, ~30 Hz):  Camera → YOLO → IBVS → Tracking Buffer
  Inner Loop (Motor, ~1000 Hz): Buffer → Interpolation → Velocity Limiting → PWM

Uses mask_area from detector.py as confidence metric for dynamic filter tuning.
Non-blocking shared state handoff via TrackingBuffer.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math


@dataclass
class VisionState:
    """One complete vision frame measurement."""
    timestamp: float
    error_x: float                # Normalized error [-1, 1]
    error_y: float                # Normalized error [-1, 1]
    mask_area: Optional[int]      # Segmented pixel count (NEW: confidence metric)
    detected: bool
    aligned: bool
    stable: bool
    confidence: float             # Detection confidence [0, 1]

    def quality(self) -> float:
        """Combined confidence: detection + segmentation area."""
        if not self.detected or self.mask_area is None:
            return 0.0

        # Normalize mask_area (typical visible area ~50k-150k pixels)
        mask_norm = min(self.mask_area / 100000, 1.0)

        # Weighted: detection (80%) + segmentation (20%)
        return 0.8 * self.confidence + 0.2 * mask_norm


@dataclass
class MotorCommand:
    """Command for chassis controller."""
    forward: float      # [-1, 1]
    steer: float        # [-1, 1]
    timestamp: float


class TrackingBuffer:
    """
    Circular buffer storing vision states.
    Non-blocking: writer (vision thread) doesn't wait for reader (motor thread).
    Uses RLock for minimal contention.
    """

    def __init__(self, max_size: int = 5):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.latest_state = None

    def push(self, state: VisionState):
        """Vision thread: push new detection (non-blocking)."""
        with self.lock:
            self.buffer.append(state)
            self.latest_state = state  # Atomic assignment (GIL)

    def snapshot(self) -> List[VisionState]:
        """Motor thread: get current buffer snapshot (low-latency)."""
        with self.lock:
            return list(self.buffer)

    def get_latest(self) -> Optional[VisionState]:
        """Direct read of newest state (minimal locking)."""
        return self.latest_state

    def estimate_velocity(self) -> Tuple[float, float]:
        """Compute (dx/dt, dy/dt) from last two frames for trajectory prediction."""
        snapshot = self.snapshot()
        if len(snapshot) < 2:
            return (0.0, 0.0)

        s_prev = snapshot[-2]
        s_curr = snapshot[-1]
        dt = s_curr.timestamp - s_prev.timestamp

        if dt < 1e-6:
            return (0.0, 0.0)

        vx = (s_curr.error_x - s_prev.error_x) / dt
        vy = (s_curr.error_y - s_prev.error_y) / dt
        return (vx, vy)

    def average_quality(self) -> float:
        """Average quality across buffer (weighted by mask_area trend)."""
        snapshot = self.snapshot()
        if not snapshot:
            return 0.0

        qualities = [s.quality() for s in snapshot]
        return sum(qualities) / len(qualities)


class AlphaBetaFilter:
    """
    Lightweight Kalman-like filter for smoothing vision updates.
    Estimates position and velocity from noisy measurements.
    Handles: frame drops, delays, jitter.
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.3):
        """
        Args:
            alpha: Position smoothing gain [0, 1]. Higher = trust measurement more.
            beta: Velocity smoothing gain [0, 1]. Higher = change velocity faster.
        """
        self.alpha = alpha
        self.beta = beta
        self.x_est = 0.0      # Estimated position
        self.v_est = 0.0      # Estimated velocity
        self.last_time = None

    def update(self, z_measured: float, timestamp: float) -> Tuple[float, float]:
        """
        Update filter with new measurement.

        Args:
            z_measured: Measured position (e.g., error_x)
            timestamp: Time of measurement

        Returns:
            (filtered_position, estimated_velocity)
        """
        if self.last_time is None:
            self.last_time = timestamp
            self.x_est = z_measured
            return (self.x_est, 0.0)

        dt = timestamp - self.last_time
        self.last_time = timestamp

        if dt < 1e-6:
            return (self.x_est, self.v_est)

        # Predict step (use last velocity)
        x_pred = self.x_est + self.v_est * dt

        # Measurement residual
        residual = z_measured - x_pred

        # Update step: adjust position and velocity based on error
        self.x_est = x_pred + self.alpha * residual
        self.v_est = self.v_est + self.beta * (residual / dt)

        return (self.x_est, self.v_est)


class TrajectoryInterpolator:
    """
    Cubic Hermite spline interpolator for smooth motion between vision updates.
    Generates continuous position/velocity curve from boundary conditions.
    """

    def __init__(self):
        self.p0 = self.p1 = 0.0  # Start and end positions
        self.v0 = self.v1 = 0.0  # Start and end velocities
        self.t_start = self.t_end = 0.0

    def set_waypoints(self, p0: float, p1: float, v0: float, v1: float,
                      t_start: float, t_end: float):
        """
        Define cubic Hermite spline with boundary conditions.

        Args:
            p0, p1: Start and end positions
            v0, v1: Start and end velocities (slopes)
            t_start, t_end: Time interval for spline
        """
        self.p0, self.p1 = p0, p1
        self.v0, self.v1 = v0, v1
        self.t_start, self.t_end = t_start, t_end

    def evaluate(self, t: float) -> Tuple[float, float]:
        """
        Evaluate cubic Hermite spline at time t.

        Returns:
            (position, velocity)
        """
        dt = self.t_end - self.t_start
        if dt < 1e-6:
            return (self.p0, self.v0)

        # Normalized time [0, 1]
        s = (t - self.t_start) / dt
        s = max(0.0, min(1.0, s))  # Clamp to [0, 1]

        # Cubic Hermite basis functions
        h00 = 2*s**3 - 3*s**2 + 1      # Position weight (start)
        h10 = s**3 - 2*s**2 + s        # Velocity weight (start)
        h01 = -2*s**3 + 3*s**2         # Position weight (end)
        h11 = s**3 - s**2              # Velocity weight (end)

        # Position interpolation
        p = (h00 * self.p0 + h10 * self.v0 * dt +
             h01 * self.p1 + h11 * self.v1 * dt)

        # Velocity interpolation (time derivative of spline)
        v = ((6*s**2 - 6*s) / dt * self.p0 +
             (3*s**2 - 4*s + 1) * self.v0 +
             (-6*s**2 + 6*s) / dt * self.p1 +
             (3*s**2 - 2*s) * self.v1)

        return (p, v)


class VelocityLimiter:
    """
    Rate limiter to prevent jerky acceleration.
    Clamps velocity change to max acceleration per timestep.
    """

    def __init__(self, max_accel: float = 0.5, dt: float = 0.001):
        """
        Args:
            max_accel: Maximum acceleration (error_units / sec^2)
            dt: Control loop timestep (1/1000 Hz = 0.001 s)
        """
        self.max_accel = max_accel
        self.dt = dt
        self.last_velocity = 0.0

    def limit(self, desired_velocity: float) -> float:
        """
        Clamp velocity to prevent acceleration overshoot.

        Args:
            desired_velocity: Unconstrained velocity from interpolator

        Returns:
            Limited velocity (smooth ramp up/down)
        """
        max_delta_v = self.max_accel * self.dt
        delta_v = desired_velocity - self.last_velocity
        delta_v = max(-max_delta_v, min(max_delta_v, delta_v))

        self.last_velocity += delta_v
        return self.last_velocity


class CascadeController:
    """
    Main coordinator: manages vision thread, motor thread, and shared buffer.
    Non-blocking architecture for real-time performance.
    """

    def __init__(self, detector, ibvs_centering, chassis_controller=None):
        """
        Args:
            detector: AluminiumCanDetector instance
            ibvs_centering: IBVSCentering controller instance
            chassis_controller: Optional ChassisController for actual motor control
        """
        self.detector = detector
        self.ibvs_centering = ibvs_centering
        self.chassis = chassis_controller

        self.buffer = TrackingBuffer(max_size=5)
        self.vision_thread = None
        self.motor_thread = None
        self.running = False

    def start(self):
        """Launch vision and motor threads."""
        self.running = True

        self.vision_thread = _VisionWorker(
            self.detector, self.ibvs_centering, self.buffer
        )
        self.motor_thread = _MotorWorker(
            self.buffer, self.chassis
        )

        self.vision_thread.daemon = True
        self.motor_thread.daemon = True

        self.vision_thread.start()
        self.motor_thread.start()

        print("✓ Cascade controller started (Vision ~30Hz, Motor ~1000Hz)")

    def stop(self):
        """Stop both threads gracefully."""
        self.running = False
        if self.vision_thread:
            self.vision_thread.join(timeout=2.0)
        if self.motor_thread:
            self.motor_thread.join(timeout=2.0)
        print("✓ Cascade controller stopped")

    def get_status(self) -> Optional[VisionState]:
        """Get latest vision state (for monitoring)."""
        return self.buffer.get_latest()


class _VisionWorker(threading.Thread):
    """Vision capture thread (~30 Hz)."""

    def __init__(self, detector, ibvs_centering, buffer: TrackingBuffer):
        super().__init__()
        self.detector = detector
        self.ibvs = ibvs_centering
        self.buffer = buffer
        self.running = False

    def run(self):
        """Main vision loop."""
        self.running = True
        frame_interval = 1.0 / 30  # 30 Hz

        while self.running:
            t_start = time.time()

            try:
                # Capture and detect
                frame = self.detector.read_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                result = self.detector.infer(frame)
                status = self.ibvs.update(result)

                # Extract mask_area from best detection
                mask_area = None
                if result.best:
                    mask_area = result.best.mask_area

                # Create vision state with mask_area
                state = VisionState(
                    timestamp=time.time(),
                    error_x=status.error_x,
                    error_y=status.error_y,
                    mask_area=mask_area,
                    detected=result.found,
                    aligned=status.aligned,
                    stable=status.stable,
                    confidence=result.best.confidence if result.best else 0.0
                )

                # Push to shared buffer (non-blocking)
                self.buffer.push(state)

            except Exception as e:
                print(f"[Vision] Error: {e}")

            # Maintain frame rate
            elapsed = time.time() - t_start
            time.sleep(max(0.001, frame_interval - elapsed))

    def stop(self):
        self.running = False


class _MotorWorker(threading.Thread):
    """Motor control thread (~1000 Hz)."""

    def __init__(self, buffer: TrackingBuffer, chassis_controller=None):
        super().__init__()
        self.buffer = buffer
        self.chassis = chassis_controller
        self.running = False

        # Filters for X and Y axes
        self.filter_x = AlphaBetaFilter(alpha=0.6, beta=0.3)
        self.filter_y = AlphaBetaFilter(alpha=0.6, beta=0.3)

        # Interpolators
        self.interp_x = TrajectoryInterpolator()
        self.interp_y = TrajectoryInterpolator()

        # Velocity limiters
        self.limiter_x = VelocityLimiter(max_accel=0.5, dt=0.001)
        self.limiter_y = VelocityLimiter(max_accel=0.5, dt=0.001)

        self.last_vision_time = 0.0

    def run(self):
        """Main motor control loop."""
        self.running = True
        dt = 1.0 / 1000  # 1000 Hz

        while self.running:
            t_start = time.time()
            t_now = t_start

            # Check for new vision data
            latest = self.buffer.get_latest()
            if latest and latest.timestamp > self.last_vision_time:
                self._handle_vision_update(latest)
                self.last_vision_time = latest.timestamp

            # Interpolate current setpoint
            cmd = self._interpolate_command(t_now)

            # Apply to motor
            self._send_motor_command(cmd)

            # Maintain loop rate
            elapsed = time.time() - t_start
            time.sleep(max(0.0001, dt - elapsed))

    def _handle_vision_update(self, state: VisionState):
        """New vision frame: update splines and filter gains."""
        t = state.timestamp

        # Get velocity estimate from buffer history
        vx, vy = self.buffer.estimate_velocity()

        # Update filter gains based on mask_area confidence
        quality = state.quality()
        alpha = 0.3 + 0.4 * quality  # Varies [0.3, 0.7]
        self.filter_x.alpha = alpha
        self.filter_y.alpha = alpha

        # Filter positions and velocities
        px_filt, vx_filt = self.filter_x.update(state.error_x, t)
        py_filt, vy_filt = self.filter_y.update(state.error_y, t)

        # Set trajectory splines for next update (33ms at 30Hz)
        t_next = t + 0.033

        self.interp_x.set_waypoints(
            px_filt, px_filt + vx_filt * 0.033,
            vx_filt, vx_filt, t, t_next
        )
        self.interp_y.set_waypoints(
            py_filt, py_filt + vy_filt * 0.033,
            vy_filt, vy_filt, t, t_next
        )

    def _interpolate_command(self, t: float) -> MotorCommand:
        """Interpolate setpoint at current time."""
        px, vx = self.interp_x.evaluate(t)
        py, vy = self.interp_y.evaluate(t)

        # Velocity limiting (prevent jerky acceleration)
        vx_lim = self.limiter_x.limit(vx)
        vy_lim = self.limiter_y.limit(vy)

        # Convert error → motor command with gain
        # Scale factor of 0.8 ensures reasonable motor speeds
        forward = -vy_lim * 0.8  # -error_y → forward
        steer = vx_lim * 0.8     # error_x → steer

        # Apply minimum speed to prevent stalling
        # If magnitude is small but non-zero, boost to minimum
        mag = math.sqrt(forward**2 + steer**2)
        if 0.01 < mag < 0.15:
            scale = 0.15 / mag
            forward *= scale
            steer *= scale

        # Clamp to [-1, 1] range
        forward = max(-1.0, min(1.0, forward))
        steer = max(-1.0, min(1.0, steer))

        return MotorCommand(forward, steer, t)

    def _send_motor_command(self, cmd: MotorCommand):
        """Send command to chassis motor driver."""
        if self.chassis:
            try:
                self.chassis.set_motor_pwm(cmd.forward, cmd.steer)
            except Exception as e:
                print(f"[Motor] Error sending command: {e}")

    def stop(self):
        self.running = False
