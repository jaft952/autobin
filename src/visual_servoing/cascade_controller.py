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

# Motor output is zeroed (and the base BRAKED) when the newest DETECTED
# vision state is older than this. Long enough to bridge 1-2 dropped YOLO
# frames without stutter-braking; short enough that a lost tin or a dead
# camera stops the base almost immediately.
DETECT_GRACE_S = 0.6

# A motor command may only act on vision data at most this old — beyond it,
# BRAKE and wait for the next detection instead of extrapolating. Without
# this cap the interpolator holds the last spline indefinitely, so the base
# kept executing a stale decision for the whole (slow) inference gap:
# observed as "lunge N steps forward, N steps back" hunting at the target.
CMD_MAX_AGE_S = 0.30


@dataclass
class VisionState:
    """One complete vision frame measurement."""
    timestamp: float
    error_x: float                # Normalized error [-1, 1]
    error_y: float                # Normalized error [-1, 1]
    mask_area: Optional[int]      # Segmented pixel count (confidence metric)
    detected: bool
    aligned: bool
    stable: bool
    confidence: float             # Detection confidence [0, 1]
    result: Optional[object] = None  # DetectionResult, untyped to avoid importing torch here

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

    def reset(self):
        """Forget slew state. Call whenever output is force-zeroed, or the
        next limit() slews FROM the stale value — a wrong-direction lurch."""
        self.last_velocity = 0.0


class CascadeController:
    """
    Main coordinator: manages vision thread, motor thread, and shared buffer.
    Non-blocking architecture for real-time performance.
    """

    def __init__(self, detector, ibvs_centering, chassis_controller=None, speed_scale: float = 1.0,
                 driving: bool = True, steer_scale: Optional[float] = None,
                 mode: str = "step", speed_min: float = 0.0, steer_min: float = 0.0):
        """
        Args:
            detector: AluminiumCanDetector instance
            ibvs_centering: IBVSCentering controller instance
            chassis_controller: Optional ChassisController for actual motor control
            speed_scale: multiplier (0..1) on the forward component — the MAX
                forward output when speed_min is also given
            speed_min: > 0 maps the forward output into [speed_min,
                speed_scale] by error magnitude — far = max, close = min, so
                the approach slows down but never below a speed that moves
                (--speed 0.6-0.3 on the test CLI)
            steer_scale: multiplier (0..1) on the steer component — turning has
                no rolling friction so it runs away at the forward scale; None
                = same as speed_scale
            driving: whether motor commands actually reach the chassis — the real
                gate; see set_driving()
            mode: "step" = pivot until the tin is central, then drive straight,
                re-aim when it drifts out of the central 70%;
                "chase" = car-like continuous pursuit (drive + steer together),
                pivoting only if the tin nears the frame edge;
                "cruise" = chase without the stale-data brake-stops: fast
                320px detection, dead-reckoning between frames, speed tapers
                with distance for a smooth continuous approach.
                All stop+brake once inside the arc-grasp zone.
        """
        self.detector = detector
        self.ibvs_centering = ibvs_centering
        self.chassis = chassis_controller
        self.speed_scale = speed_scale
        self.speed_min = min(speed_min, speed_scale)
        self.steer_scale = speed_scale if steer_scale is None else steer_scale
        self.steer_min = min(steer_min, self.steer_scale)
        self.mode = mode
        self.driving = driving

        self.buffer = TrackingBuffer(max_size=5)
        self.vision_thread = None
        self.motor_thread = None
        self.running = False

    def set_driving(self, driving: bool):
        """Enable/disable actual motor output. Safe to call from any thread."""
        self.driving = driving
        if self.motor_thread is not None:
            self.motor_thread.driving = driving

    def set_mode(self, mode: str):
        """Switch pursuit mode live: "step" or "chase"."""
        self.mode = mode
        if self.motor_thread is not None:
            self.motor_thread.mode = mode

    def start(self):
        """Launch vision and motor threads."""
        self.running = True

        self.vision_thread = _VisionWorker(
            self.detector, self.ibvs_centering, self.buffer,
            get_mode=lambda: self.mode
        )
        self.motor_thread = _MotorWorker(
            self.buffer, self.chassis, speed_scale=self.speed_scale,
            steer_scale=self.steer_scale, driving=self.driving, mode=self.mode,
            speed_min=self.speed_min, steer_min=self.steer_min
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

    def get_last_command(self) -> Optional[MotorCommand]:
        """Get last motor command sent (for display/logging)."""
        return self.motor_thread.last_cmd if self.motor_thread else None

    def get_last_sent(self) -> Tuple[float, float]:
        """Actual (forward, steer) values last sent to the chassis, after
        speed_min/speed_scale/steer_scale mapping — for on-screen display."""
        return self.motor_thread.last_sent if self.motor_thread else (0.0, 0.0)


class _VisionWorker(threading.Thread):
    """Vision capture thread (~30 Hz)."""

    def __init__(self, detector, ibvs_centering, buffer: TrackingBuffer,
                 get_mode=None):
        super().__init__()
        self.detector = detector
        self.ibvs = ibvs_centering
        self.buffer = buffer
        self.get_mode = get_mode or (lambda: "step")
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

                # cruise trades accuracy for detection RATE: 320px inference
                # (auto-falls back if the model backend rejects it) and no
                # orientation estimation. step/chase keep the full pipeline.
                if self.get_mode() == "cruise":
                    result = self.detector.infer(frame, imgsz=320, fast=True)
                else:
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
                    confidence=result.best.confidence if result.best else 0.0,
                    result=result,
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

    def __init__(self, buffer: TrackingBuffer, chassis_controller=None, speed_scale: float = 1.0,
                 driving: bool = True, steer_scale: Optional[float] = None,
                 mode: str = "step", speed_min: float = 0.0, steer_min: float = 0.0):
        super().__init__()
        self.buffer = buffer
        self.chassis = chassis_controller
        self.speed_scale = speed_scale
        self.speed_min = min(speed_min, speed_scale)
        self.steer_scale = speed_scale if steer_scale is None else steer_scale
        self.steer_min = min(steer_min, self.steer_scale)
        self.last_sent: Tuple[float, float] = (0.0, 0.0)
        self.driving = driving  # actual gate on motor output
        self._was_driving = driving
        self.mode = mode        # "step" | "chase" | "cruise" (see CascadeController)
        self._aiming = False    # pivot-in-place state (both modes use it)
        self._last_good: Optional[VisionState] = None  # newest DETECTED state
        self._moving = False    # last command was nonzero -> brake on stop
        self.running = False

        # Filters for X and Y axes
        self.filter_x = AlphaBetaFilter(alpha=0.6, beta=0.3)
        self.filter_y = AlphaBetaFilter(alpha=0.6, beta=0.3)

        # Interpolators
        self.interp_x = TrajectoryInterpolator()
        self.interp_y = TrajectoryInterpolator()

        # Slew limiters on the motor command (2.0/s: full output in ~0.4 s —
        # smooths spikes/flips without making the P-control feel laggy)
        self.limiter_x = VelocityLimiter(max_accel=2.0, dt=0.001)
        self.limiter_y = VelocityLimiter(max_accel=2.0, dt=0.001)

        self.last_vision_time = 0.0
        self.last_cmd: Optional[MotorCommand] = None

    def run(self):
        """Main motor control loop."""
        self.running = True
        dt = 1.0 / 1000  # 1000 Hz

        while self.running:
            t_start = time.time()
            t_now = t_start

            # Check for new vision data. Only DETECTED states feed the
            # filters — undetected frames carry error 0,0 and would drag the
            # splines toward "target centered" during a dropout.
            latest = self.buffer.get_latest()
            if (latest and latest.detected
                    and latest.timestamp > self.last_vision_time):
                self._handle_vision_update(latest)
                self.last_vision_time = latest.timestamp
                self._last_good = latest

            # Gate. DETECT_GRACE_S bridges 1-2 dropped detections so the base
            # doesn't stutter-brake through YOLO flicker; anything older (tin
            # gone, camera dead) or already aligned -> zero, and
            # _send_motor_command turns that into an active BRAKE, not a coast
            # (coasting at speed is what kept overshooting the grasp zone).
            # cruise keeps rolling between detections (dead-reckons on the
            # spline + tapers speed with distance) instead of brake-waiting.
            cmd_aged = ((t_now - self.last_vision_time) > CMD_MAX_AGE_S
                        and self.mode != "cruise")
            good = self._last_good
            if (good is None or (t_now - good.timestamp) > DETECT_GRACE_S
                    or good.aligned or cmd_aged):
                # Lost/aligned -> stop; data merely AGED -> brake and wait for
                # the next detection rather than acting on a stale decision.
                cmd = MotorCommand(0.0, 0.0, t_now)
                self.limiter_x.reset()   # else the next command slews from a
                self.limiter_y.reset()   # stale value = random-direction lurch
            else:
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

        # Spline horizon: cruise dead-reckons the error trend across the real
        # inference gap so the base can keep rolling between detections;
        # step/chase only bridge one nominal frame.
        horizon = 0.30 if self.mode == "cruise" else 0.033
        t_next = t + horizon

        self.interp_x.set_waypoints(
            px_filt, px_filt + vx_filt * horizon,
            vx_filt, vx_filt, t, t_next
        )
        self.interp_y.set_waypoints(
            py_filt, py_filt + vy_filt * horizon,
            vy_filt, vy_filt, t, t_next
        )

    def _interpolate_command(self, t: float) -> MotorCommand:
        """P-control on the filtered position error.

        Signs follow the hardware-verified set_motor_pwm convention from
        test_differential_drive CASC mode: positive forward = robot forward,
        positive steer = turn left. error_y + = too close -> back up;
        error_x + = tin right of center -> turn right. Hence both negated.
        (The old version drove on the error's VELOCITY estimate, so with a
        static tin the command was mostly filter noise boosted to MIN_SPEED —
        random-looking spins.)"""
        px, _vx = self.interp_x.evaluate(t)
        py, _vy = self.interp_y.evaluate(t)

        FWD_GAIN = 1.5
        STEER_GAIN = 1.0   # yaw overshoots hard at vision rate — keep gentler than forward

        # Pivot-in-place ("aiming") applies in BOTH modes, with different entry:
        #   step  : tin outside the central 70% of the frame -> re-aim
        #   chase : only when the tin nears the frame EDGE (about to be lost)
        # Hysteresis so the state doesn't chatter at the boundary. Entering
        # aim returns a ZERO command for this tick — _send_motor_command turns
        # that into a BRAKE, so the base stops its lunge BEFORE pivoting.
        AIM_ENTER = 0.35 if self.mode == "step" else 0.42
        AIM_EXIT = 0.25
        if self._aiming:
            if abs(px) <= AIM_EXIT:
                self._aiming = False
        elif abs(px) >= AIM_ENTER:
            self._aiming = True
            self.limiter_x.reset()
            self.limiter_y.reset()
            return MotorCommand(0.0, 0.0, t)   # brake first, pivot next tick

        if self._aiming:
            forward = self.limiter_y.limit(0.0)
            steer = self.limiter_x.limit(-px * STEER_GAIN)
        elif self.mode == "step":
            # step: drive straight; lateral drift is handled by re-aiming.
            forward = self.limiter_y.limit(-py * FWD_GAIN)
            steer = self.limiter_x.limit(0.0)
        else:
            # chase/cruise: car-like — keep rolling, steer while moving.
            # (P on distance already tapers cruise speed as the tin nears.)
            forward = self.limiter_y.limit(-py * FWD_GAIN)
            steer = self.limiter_x.limit(-px * STEER_GAIN)

        # MIN duty floor (stall avoidance) with a true deadband under it.
        # 0.50 boosted every small correction to a half-speed lunge -> ±0.4
        # error_x limit cycle around the target; 0.35 still beats stall.
        # cruise floors LOWER so the taper can actually slow the approach —
        # raise it if the base stalls while creeping in.
        mag = math.sqrt(forward**2 + steer**2)
        if self.mode == "cruise":
            # With an explicit speed_min the [min,max] output mapping is the
            # floor — keep only a tiny one here so the taper can reach min.
            MIN_SPEED = 0.05 if self.speed_min > 0 else 0.22
        else:
            MIN_SPEED = 0.35
        if 0.01 < mag < MIN_SPEED:
            scale = MIN_SPEED / mag
            forward *= scale
            steer *= scale
        elif mag < 0.01:
            forward = 0.0
            steer = 0.0

        forward = max(-1.0, min(1.0, forward))
        steer = max(-1.0, min(1.0, steer))

        return MotorCommand(forward, steer, t)

    def _brake(self):
        """Active hold (shorted windings) — a coast at approach speed rolls
        right past the grasp zone; that overshoot was observed on the robot."""
        actuator = getattr(self.chassis, "actuator", None)
        brake = getattr(actuator, "brake", None)
        if brake is not None:
            brake()
        else:
            self.chassis.stop()

    def _send_motor_command(self, cmd: MotorCommand):
        """Send command to chassis motor driver — gated by self.driving.
        A zero command BRAKES once on the moving->stopped transition, then
        goes quiet (no 1 kHz zero-PWM spam)."""
        self.last_cmd = cmd  # Track for display/logging
        if not self.chassis:
            return
        try:
            if self.driving:
                if abs(cmd.forward) < 0.01 and abs(cmd.steer) < 0.01:
                    if self._moving:
                        self._brake()
                        self._moving = False
                    self.last_sent = (0.0, 0.0)
                else:
                    # Forward output: plain scale, or — when speed_min is set —
                    # mapped into [speed_min, speed_scale] by command magnitude
                    # so the approach tapers but never crawls below speed_min.
                    if self.speed_min > 0 and abs(cmd.forward) >= 0.01:
                        fwd_out = math.copysign(
                            self.speed_min + (self.speed_scale - self.speed_min)
                            * min(1.0, abs(cmd.forward)),
                            cmd.forward)
                    else:
                        fwd_out = cmd.forward * self.speed_scale
                    if self.steer_min > 0 and abs(cmd.steer) >= 0.01:
                        steer_out = math.copysign(
                            self.steer_min + (self.steer_scale - self.steer_min)
                            * min(1.0, abs(cmd.steer)),
                            cmd.steer)
                    else:
                        steer_out = cmd.steer * self.steer_scale
                    self.chassis.set_motor_pwm(fwd_out, steer_out)
                    self.last_sent = (fwd_out, steer_out)
                    self._moving = True
            elif self._was_driving:
                self.chassis.stop()  # actively cut power once, not just stop sending
                self._moving = False
                self.last_sent = (0.0, 0.0)
            self._was_driving = self.driving
        except Exception as e:
            print(f"[Motor] Error sending command: {e}")

    def stop(self):
        self.running = False
