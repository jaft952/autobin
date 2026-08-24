"""
web/runtime.py

RobotRuntime — the robot's main loop as a controllable background thread,
driven by the web dashboard (web/server.py).

It owns exactly what tests/test_subsumption_live.py wires up by hand:

    SensorHub(camera + ultrasonic) -> layers -> Arbitrator
        -> MotionExecutor (wheels) + ArmExecutor (arm)

but adds an operating-state machine on top:

    STOPPED   sensors tick (camera preview, distance, vitals stay live),
              wheels held stopped, layers not evaluated. Manual arm allowed.
    AUTO      full stack: zigzag patrol -> approach -> collect.
    SCAN      "manual scanning": zigzag patrol only (layers 0/1/5), the
              camera layers are left out even if a camera is present.
    ESTOP     wheels forced stopped every tick until the operator restarts.

E-STOP nuance: estop() also cuts motor PWM immediately from the caller's
thread — it does not wait for the loop tick, because a blocking grab can
hold the loop for seconds. A grab in progress cannot be interrupted
(stepped servo moves are open-loop); the wheels are already halted during
grabs by design.

Settings changed via the dashboard patch module/class attributes live
(layers read them every tick). They are session-only — edit the source for
permanent values.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor, UltrasonicPins
from src.hardware.sensors.sensor_hub import SensorHub
from src.subsumption.arbitrator import Arbitrator
from src.subsumption.motion_executor import MotionExecutor
from src.subsumption.layers.layer0_idle import SystemIdleLayer
import src.subsumption.layers.layer1_scan as scan_mod
from src.subsumption.layers.layer1_scan import ScanAroundLayer
from src.subsumption.layers.layer2_approach import ApproachLitterLayer
from src.subsumption.layers.layer3_collect import CollectLitterLayer
from src.subsumption.layers.layer5_emergency import EmergencyStopLayer

STATE_STOPPED = "STOPPED"
STATE_AUTO = "AUTO"
STATE_SCAN = "SCAN"
STATE_ESTOP = "ESTOP"

FRAME_ENCODE_EVERY = 2          # encode the camera jpeg every Nth tick
PERF_WINDOW = 50                # ticks averaged for the performance card


class RobotRuntime:

    def __init__(self, log, with_camera: bool = True, hz: float = 20.0):
        self.log = log
        self.hz = float(hz)
        self._state = STATE_STOPPED
        self._state_lock = threading.Lock()
        self._arm_lock = threading.Lock()

        # ── Sensors (camera is best-effort: missing torch/model/webcam just
        #    downgrades the dashboard, it must never kill the server) ──────
        camera = None
        if with_camera:
            try:
                from src.hardware.sensors.camera_sensor import CameraSensor
                camera = CameraSensor()
            except Exception as exc:
                self.log.warning(f"camera unavailable ({exc}) — running without it")
        battery = None
        try:
            from src.hardware.sensors.battery_sensor import BatterySensor
            battery = BatterySensor()
        except Exception as exc:
            self.log.warning(f"battery sensor unavailable ({exc}) — using placeholder")
        self.sensors = SensorHub(
            front=UltrasonicSensor(UltrasonicPins(trig=23, echo=24)),
            back=UltrasonicSensor(UltrasonicPins(trig=17, echo=20)),
            front_left=UltrasonicSensor(UltrasonicPins(trig=27, echo=22)),
            front_right=UltrasonicSensor(UltrasonicPins(trig=5, echo=6)),
            camera=camera,
            battery=battery,
        )
        self.camera_available = camera is not None

        # ── Layers / arbitration / executors ─────────────────────────────
        idle_layer = SystemIdleLayer()
        self.scan_layer = ScanAroundLayer()
        self.emergency_layer = EmergencyStopLayer()
        collect_layer = CollectLitterLayer()
        # Both mode lists share the SAME layer objects. Separate copies meant
        # tuning done in SCAN was lost on the way to AUTO, and the unused copy
        # came back mid-manoeuvre with a phase timer from minutes earlier.
        self._layers_scan = [idle_layer, self.scan_layer, self.emergency_layer]
        self._layers_auto = [idle_layer, self.scan_layer, ApproachLitterLayer(),
                             collect_layer, self.emergency_layer]
        self._all_layers = self._layers_auto   # superset of _layers_scan
        self.arbitrator = Arbitrator()
        self.motion = MotionExecutor()

        self.arm = None
        try:
            from src.subsumption.arm_executor import ArmExecutor
            # Does NOT move the arm at server boot — the first arm command
            # after the operator presses START force-homes it.
            self.arm = ArmExecutor(
                sensors=self.sensors if self.camera_available else None,
                grab_zone_check=collect_layer.can_still_in_grab_zone,
            )
        except Exception as exc:
            self.log.warning(f"arm unavailable ({exc}) — manual arm + grabs disabled")

        # ── Telemetry ─────────────────────────────────────────────────────
        self.started_at = time.monotonic()
        self.win_message = ""
        self.win_layer = -1
        self._tick_times = deque(maxlen=PERF_WINDOW)
        self._tick_stamps = deque(maxlen=PERF_WINDOW)
        self._frame_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._stream_clients = 0              # MJPEG viewers; 0 -> skip encoding
        self._stream_lock = threading.Lock()
        self._cpu_prev = None                 # (idle, total) from /proc/stat

        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="robot-loop")

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        self.sensors.start()
        self._thread.start()
        self.log.info("runtime loop started (state STOPPED)")

    def close(self):
        """Full teardown: stop loop thread, motors, GPIO, camera."""
        self._alive = False
        self._thread.join(timeout=2.0)
        try:
            self.motion.close()
        finally:
            self.sensors.stop()
        self.log.info("runtime closed (GPIO released)")

    # ── State transitions (called from Flask request threads) ────────────

    @property
    def state(self) -> str:
        return self._state

    def start_auto(self):
        self._transition(STATE_AUTO, "system START (full autonomy)")

    def start_scan(self):
        self._transition(STATE_SCAN, "manual SCANNING (zigzag patrol only)")

    def stop(self):
        self._transition(STATE_STOPPED, "system STOP")
        self.motion.stop()

    def estop(self):
        # Cut power NOW from this thread — the loop may be inside a blocking
        # grab and must not be waited on.
        self.motion.stop()
        self._transition(STATE_ESTOP, "EMERGENCY STOP")

    def _transition(self, new_state: str, why: str):
        with self._state_lock:
            old, self._state = self._state, new_state
        # SCAN-only has no approach layer, so the patrol must keep driving
        # past cans instead of standing down for them.
        self.scan_layer.yield_to_targets = (new_state == STATE_AUTO)
        for layer in self._all_layers:
            layer.reset()
        self.log.warning(f"{why}  [{old} -> {new_state}]")
        self.win_message, self.win_layer = "", -1

    # ── Main loop ─────────────────────────────────────────────────────────

    def _run(self):
        tick_n = 0
        while self._alive:
            t0 = time.monotonic()
            try:
                self._tick(tick_n)
            except Exception as exc:
                # One bad tick must not kill the robot thread.
                self.log.error(f"tick error: {exc!r}")
            tick_n += 1

            self._tick_times.append(time.monotonic() - t0)
            self._tick_stamps.append(t0)
            sleep_left = (1.0 / self.hz) - (time.monotonic() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)

    def _tick(self, tick_n: int):
        self.sensors.update()

        state = self._state
        if state in (STATE_AUTO, STATE_SCAN):
            layers = self._layers_auto if state == STATE_AUTO else self._layers_scan
            for layer in layers:
                self.arbitrator.submit_command(layer.evaluate(self.sensors))
            winning = self.arbitrator.get_winning_action()
            for layer in layers:
                layer.notify_arbitration(layer.layer_id == winning.layer_id)
            self.arbitrator.clear()

            self.motion.execute(winning)
            if self.arm is not None:
                with self._arm_lock:
                    self.arm.execute(winning)

            if winning.message != self.win_message:
                self.log.info(f"[L{winning.layer_id}] {winning.message}")
                if winning.layer_id == 0:
                    self.log.warning("no layer wants to drive - base parked on IDLE")
            self.win_message, self.win_layer = winning.message, winning.layer_id
        else:
            # STOPPED / ESTOP: enforce halted wheels every tick.
            self.motion.stop()

        # Annotating + JPEG-encoding a frame costs real CPU on the Pi — only
        # pay it while someone is actually watching the camera stream.
        if (self.camera_available and self._stream_clients > 0
                and tick_n % FRAME_ENCODE_EVERY == 0):
            self._encode_frame()

    def _encode_frame(self):
        try:
            import cv2
            frame = self.sensors.get_annotated_frame()
            if frame is None:
                return
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._frame_lock:
                    self._frame_jpeg = jpeg.tobytes()
        except Exception:
            pass                              # never let preview kill the loop

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._frame_jpeg

    def stream_client_connected(self):
        with self._stream_lock:
            self._stream_clients += 1

    def stream_client_disconnected(self):
        with self._stream_lock:
            self._stream_clients = max(0, self._stream_clients - 1)

    # ── Manual arm control (STOPPED only) ─────────────────────────────────

    def _manual_arm_planner(self):
        if self.arm is None:
            raise RuntimeError("arm hardware not available")
        if self._state != STATE_STOPPED:
            raise RuntimeError(f"manual arm only in STOPPED state (now {self._state})")
        return self.arm.planner

    def arm_pose(self) -> Optional[dict]:
        if self.arm is None:
            return None
        return self.arm.planner.get_pose()

    def arm_named_pose(self, name: str):
        with self._arm_lock:
            planner = self._manual_arm_planner()
            if name == "home":
                # goto() asserts every channel, so this works even when the
                # tracked pose already claims home (e.g. right after boot).
                planner.goto("home")
                planner.open_gripper()
                self.arm._at_home = True
                self.arm._force_next_home = False
            elif name == "bin":
                planner.dump_to_bin()       # release whatever is held
                self.arm._at_home = False
            else:
                raise ValueError(f"unknown pose '{name}' (home|bin)")
        self.log.info(f"manual arm -> pose '{name}'")

    def arm_gripper(self, action: str):
        with self._arm_lock:
            planner = self._manual_arm_planner()
            if action == "close":
                planner.close_gripper()
            elif action in ("open", "neutral", "stow"):
                planner.open_gripper()
            else:
                raise ValueError(f"unknown gripper action '{action}'")
        self.log.info(f"manual gripper -> {action}")

    def arm_jog(self, channel: int, delta: float) -> float:
        with self._arm_lock:
            planner = self._manual_arm_planner()
            new_val = planner.jog_channel(channel, delta)
            self.arm._at_home = False
        self.log.info(f"manual jog CH{channel + 1} {delta:+.1f} -> {new_val:.1f}")
        return new_val

    # ── Quick settings ────────────────────────────────────────────────────

    def settings_registry(self):
        """Whitelisted live-tunable parameters:
        (key, [(obj, attr), ...], lo, hi, step, label). Targets are the LIVE
        layer objects -- the layers copy the module defaults in __init__, so
        patching the module afterwards changed nothing."""
        scan, emerg = self.scan_layer, self.emergency_layer
        return [
            ("scan.forward_speed", [(scan, "forward_speed")], 0.2, 1.0, 0.05, "Scan: lane speed (0-1)"),
            # Layer 5 outvotes the scan pivot, so both must pivot at one speed.
            ("scan.turn_speed",    [(scan, "turn_speed"), (emerg, "turn_speed")],
             0.2, 1.0, 0.05, "Scan: pivot speed (0-1)"),
            ("scan.turn_90_s",     [(scan, "turn_90_s")], 0.3, 3.0, 0.05, "Scan: 90° pivot time (s)"),
            ("scan.shift_s",       [(scan, "shift_s")], 0.3, 4.0, 0.10, "Scan: lane shift time (s)"),
            ("scan.max_lane_s",    [(scan, "max_lane_s")], 3.0, 60.0, 1.0, "Scan: lane timeout (s)"),
            # Read from the module every tick, so patching the global works.
            ("scan.turn_at_cm",    [(scan_mod, "TURN_AT_CM")], 15.0, 100.0, 1.0, "Scan: turn at wall (cm)"),
            ("safety.estop_cm",    [(SensorHub, "EMERGENCY_STOP_CM")], 5.0, 30.0, 1.0, "Emergency stop range (cm)"),
            ("loop.hz",            [(self, "hz")], 2.0, 20.0, 1.0, "Control loop rate (Hz)"),
        ]

    def get_settings(self) -> list:
        return [{"key": k, "value": round(float(getattr(*targets[0])), 3),
                 "min": lo, "max": hi, "step": step, "label": label}
                for k, targets, lo, hi, step, label in self.settings_registry()]

    def set_setting(self, key: str, value: float) -> float:
        for k, targets, lo, hi, _step, _label in self.settings_registry():
            if k == key:
                clamped = max(lo, min(hi, float(value)))
                for obj, attr in targets:
                    setattr(obj, attr, clamped)
                self.log.info(f"setting {key} = {clamped} (session only)")
                return clamped
        raise KeyError(f"unknown setting '{key}'")

    # ── Status / vitals ───────────────────────────────────────────────────

    def status(self) -> dict:
        hz_actual = 0.0
        if len(self._tick_stamps) >= 2:
            span = self._tick_stamps[-1] - self._tick_stamps[0]
            if span > 0:
                hz_actual = (len(self._tick_stamps) - 1) / span
        tick_ms = (sum(self._tick_times) / len(self._tick_times) * 1000.0
                   if self._tick_times else 0.0)

        dist = self.sensors.get_obstacle_distance_cm()
        return {
            "state": self._state,
            "message": self.win_message,
            "layer": self.win_layer,
            "distance_cm": round(dist, 1) if dist is not None else None,
            "battery": round(self.sensors.get_battery_level(), 2),
            "battery_placeholder": not self.sensors.has_real_battery_sensor(),
            "camera": self.camera_available,
            "arm": self.arm is not None,
            "uptime_s": int(time.monotonic() - self.started_at),
            "loop_hz": round(hz_actual, 1),
            "tick_ms": round(tick_ms, 1),
            "cpu_pct": self._cpu_percent(),
            "mem_pct": self._mem_percent(),
            "temp_c": self._cpu_temp(),
        }

    # Pi vitals via /proc and /sys — None on platforms without them (the UI
    # shows a dash). No psutil dependency.

    def _cpu_percent(self):
        try:
            with open("/proc/stat") as f:
                parts = f.readline().split()[1:]
            vals = [int(p) for p in parts]
            idle, total = vals[3] + vals[4], sum(vals)
            prev, self._cpu_prev = self._cpu_prev, (idle, total)
            if prev is None or total == prev[1]:
                return None
            d_idle, d_total = idle - prev[0], total - prev[1]
            return round(100.0 * (1.0 - d_idle / d_total), 1)
        except Exception:
            return None

    @staticmethod
    def _mem_percent():
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.strip().split()[0])
            return round(100.0 * (1.0 - info["MemAvailable"] / info["MemTotal"]), 1)
        except Exception:
            return None

    @staticmethod
    def _cpu_temp():
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            return None
