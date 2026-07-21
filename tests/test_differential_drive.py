"""
Manual keyboard drive test — exercises each layer of the motion stack
so a fault can be pinned to the layer where it first appears.

Layers (switch with number keys):
  1 : RAW WHEELS  — hand-built WheelCommand -> PWMActuator.apply()
                    (bypasses kinematics; trims NOT applied; calibration
                     invert_left/right + swap_left_right still apply)
  2 : KINEMATICS  — DifferentialKinematics moves -> PWMActuator.apply()
                    (same path as discrete ChassisMove; trims applied)
  3 : CASCADE     — ChassisController.set_motor_pwm(forward, steer)
                    (exact path the cascade controller uses)

If a motion works in layer 1 but not layer 2/3 with the same printed wheel
values, the bug is in that upper layer. If it fails in layer 1 too, it's
pwm_driver, wiring, or power.

Drive keys (same in every mode):
  w/s : forward/backward      a/d : turn left/right
  q/e : arc fwd-left/right    z/c : arc back-left/right
  l/r : LEFT/RIGHT wheel only (finds a weak or dead motor;
        note swap_left_right still applies, so "LEFT" may be the
        other physical wheel — that itself is useful info)
  space : stop (COAST — wheels left free to spin)
  f     : BRAKE (active electrical hold — windings shorted, wheels
          resist being pushed; the same hold used during arm grabs)
  x     : quit

MOVEMENT STYLE — starts in JOG:
  JOG  (default): a tap drives for a short burst (~0.30 s) then auto-stops,
                  so one key press = one small nudge.
  HOLD (press j): a key sets the motion and it RUNS until you press space/f.
                  Use this for sustained runs / watching steady behaviour.
  j    : toggle JOG <-> HOLD
  ] /[ : lengthen / shorten the JOG burst (0.05 s steps)

Switch modes with the NUMBER keys 1 / 2 / 3 (not i/k — those are below).

Tuning keys:
  +/- : speed up/down (0.05 steps)
  i   : invert forward polarity  (only affects modes 1 & 3; mode 2 uses the
        kinematics' own calibrated directions and ignores this)
  k   : invert steer polarity    (only affects modes 1 & 3, same reason)
  b   : toggle reverse-protection — inserts a short stop before any wheel
        flips direction. If stalling/buzzing goes away with this ON, the
        cause is current spikes from instant direction flips (power/driver,
        not code).
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motion.differential_kinematics import WheelCommand


# ----------------------------------------------------------------------
# Cross-platform non-blocking single-key reader
# ----------------------------------------------------------------------
if os.name == "nt":
    import msvcrt

    def get_key(timeout: float = 0.1):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    return ch.decode()
                except UnicodeDecodeError:
                    return None
            time.sleep(0.01)
        return None
else:
    import termios
    import tty
    import select

    def get_key(timeout: float = 0.1):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            if r:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# Key -> (name, forward_intent, steer_intent)
# forward: +1 = "toward the tin", steer: -1 = left, +1 = right
KEY_MAP = {
    "w": ("FORWARD", 1.0, 0.0),
    "s": ("BACKWARD", -1.0, 0.0),
    "a": ("TURN LEFT", 0.0, -1.0),
    "d": ("TURN RIGHT", 0.0, 1.0),
    "q": ("ARC FWD-LEFT", 1.0, -0.5),
    "e": ("ARC FWD-RIGHT", 1.0, 0.5),
    "z": ("ARC BACK-LEFT", -1.0, -0.5),
    "c": ("ARC BACK-RIGHT", -1.0, 0.5),
}

MODE_NAMES = {1: "RAW ", 2: "KIN ", 3: "CASC"}


def main():
    from src.visual_servoing.chassis_controller import ChassisController

    try:
        chassis = ChassisController()
    except Exception as exc:
        print(f"[!] chassis not available: {exc}")
        return

    actuator = chassis.actuator
    kin = chassis.kin
    cal = chassis.cal

    mode = 1            # start at the lowest layer
    speed = 0.50        # command magnitude (0..1)
    fwd_sign = 1.0      # w = forward (flip live with 'i' if wiring changes)
    steer_sign = -1.0   # a = left (flip live with 'k')
    protect = False     # reverse-protection off by default (to reproduce bug)

    style = "jog"      # "held": drive while key repeats, stop on release. "jog": tap = short burst then stop.
    jog_s = 0.30                 # jog burst seconds (tune with [ / ])
    release_timeout = 0.40       # held: stop this long after the key stops

    last_wheels = [0.0, 0.0]     # last applied left/right (for flip detection)
    drive_state = {"last": 0.0, "moving": False}   # held-style bookkeeping

    print(__doc__)
    print(
        f"mode={MODE_NAMES[mode]} speed={speed:.2f} "
        f"fwd_sign={fwd_sign:+.0f} steer_sign={steer_sign:+.0f} "
        f"protect={'ON' if protect else 'OFF'} "
        f"style={style}" + (f" ({jog_s:.2f}s)" if style == "jog" else "")
    )
    print(
        f"calibration: forward_speed={cal.forward_speed} "
        f"invert_left={cal.invert_left} invert_right={cal.invert_right} "
        f"swap_left_right={cal.swap_left_right}"
    )
    print("ready — press keys to drive.\n")

    def guard_reverse(new_left: float, new_right: float):
        """Brief stop before any wheel flips direction (if protection ON)."""
        if not protect:
            return
        flip = (new_left * last_wheels[0] < 0) or (new_right * last_wheels[1] < 0)
        if flip:
            actuator.stop()
            time.sleep(0.08)

    def remember(new_left: float, new_right: float):
        last_wheels[0] = new_left
        last_wheels[1] = new_right

    def do_stop():
        actuator.stop()
        remember(0.0, 0.0)
        drive_state["moving"] = False
        print("STOP (coast — wheels free)")

    def do_brake():
        actuator.brake()  # hardware brake state, applies regardless of mode
        remember(0.0, 0.0)
        drive_state["moving"] = False
        print("BRAKE (active hold — try pushing the robot, it should resist)")

    def after_drive():
        """jog: run jog_s then auto-stop. held: mark moving; idle_check() stops it on release."""
        if style == "jog":
            time.sleep(jog_s)
            actuator.stop()
            remember(0.0, 0.0)
        else:
            drive_state["last"] = time.time()
            drive_state["moving"] = True

    def idle_check():
        """held-style: stop shortly after the key stops arriving (release)."""
        if (style == "held" and drive_state["moving"]
                and time.time() - drive_state["last"] > release_timeout):
            actuator.stop()
            remember(0.0, 0.0)
            drive_state["moving"] = False

    try:
        while True:
            key = get_key(timeout=0.1)
            if key is None:
                idle_check()          # held-style: stop after key release
                continue
            key = key.lower()

            # ---------- control keys ----------
            if key == "x":
                break
            if key in ("1", "2", "3"):
                mode = int(key)
                do_stop()
                print(f"\n[mode] {MODE_NAMES[mode]} "
                      f"({'raw wheels' if mode == 1 else 'kinematics+trims' if mode == 2 else 'set_motor_pwm cascade path'})\n")
                continue
            if key in ("+", "="):
                speed = min(1.0, speed + 0.05)
                print(f"[speed] {speed:.2f}")
                continue
            if key in ("-", "_"):
                speed = max(0.05, speed - 0.05)
                print(f"[speed] {speed:.2f}")
                continue
            if key == "i":
                fwd_sign = -fwd_sign
                print(f"[polarity] forward sign = {fwd_sign:+.0f}")
                continue
            if key == "k":
                steer_sign = -steer_sign
                print(f"[polarity] steer sign = {steer_sign:+.0f}")
                continue
            if key == "b":
                protect = not protect
                print(f"[protect] reverse-protection {'ON' if protect else 'OFF'}")
                continue
            if key == "j":
                style = "jog" if style == "held" else "held"
                do_stop()
                print(f"[style] {'JOG — a tap drives ' + format(jog_s, '.2f') + 's then stops' if style == 'jog' else 'HELD — drive while key held, stop on release'}")
                continue
            if key in ("]", "}"):
                jog_s = min(2.0, jog_s + 0.05)
                print(f"[jog] burst = {jog_s:.2f}s")
                continue
            if key in ("[", "{"):
                jog_s = max(0.05, jog_s - 0.05)
                print(f"[jog] burst = {jog_s:.2f}s")
                continue
            if key == " ":
                do_stop()
                continue
            if key == "f":
                do_brake()
                continue

            # ---------- single wheel tests ----------
            if key in ("l", "r"):
                mag = fwd_sign * cal.forward_speed * speed
                left = mag if key == "l" else 0.0
                right = mag if key == "r" else 0.0
                guard_reverse(left, right)
                actuator.apply(WheelCommand(left, right, False, "forward"))
                remember(left, right)
                print(f"[wheel] {'LEFT' if key == 'l' else 'RIGHT'} only  "
                      f"L={left:+.1f} R={right:+.1f}  "
                      f"(swap_left_right={cal.swap_left_right} may swap physical side)")
                after_drive()
                continue

            if key not in KEY_MAP:
                continue

            name, f_intent, s_intent = KEY_MAP[key]

            # ---------- layer dispatch ----------
            if mode == 1:
                # RAW: same wheel mixing as set_motor_pwm, applied directly.
                forward = fwd_sign * f_intent * speed
                steer = steer_sign * s_intent * speed
                left = cal.forward_speed * max(-1.0, min(1.0, forward + steer))
                right = cal.forward_speed * max(-1.0, min(1.0, forward - steer))
                guard_reverse(left, right)
                actuator.apply(WheelCommand(left, right, False, "forward"))
                remember(left, right)
                print(f"[RAW ] {name:14} L={left:+6.1f} R={right:+6.1f} (speed={speed:.2f})")

            elif mode == 2:
                # KINEMATICS: discrete calibrated moves, scaled by speed.
                cmd = {
                    "FORWARD": kin.forward,
                    "BACKWARD": kin.backward,
                    "TURN LEFT": kin.turn_left,
                    "TURN RIGHT": kin.turn_right,
                    "ARC FWD-LEFT": kin.arc_forward_left,
                    "ARC FWD-RIGHT": kin.arc_forward_right,
                    "ARC BACK-LEFT": kin.arc_backward_left,
                    "ARC BACK-RIGHT": kin.arc_backward_right,
                }[name]()
                cmd.left_speed *= speed
                cmd.right_speed *= speed
                guard_reverse(cmd.left_speed, cmd.right_speed)
                actuator.apply(cmd)
                remember(cmd.left_speed, cmd.right_speed)
                print(f"[KIN ] {name:14} L={cmd.left_speed:+6.1f} R={cmd.right_speed:+6.1f} "
                      f"trim={cmd.apply_trim}/{cmd.trim_set} (speed={speed:.2f})")

            elif mode == 3:
                # CASCADE: exact call the cascade controller makes.
                forward = fwd_sign * f_intent * speed
                steer = steer_sign * s_intent * speed
                # mirror set_motor_pwm's wheel math for flip detection / printout
                left = cal.forward_speed * max(-1.0, min(1.0, forward + steer))
                right = cal.forward_speed * max(-1.0, min(1.0, forward - steer))
                if protect:
                    flip = (left * last_wheels[0] < 0) or (right * last_wheels[1] < 0)
                    if flip:
                        chassis.stop()
                        time.sleep(0.08)
                chassis.set_motor_pwm(forward, steer)
                remember(left, right)
                print(f"[CASC] {name:14} fwd={forward:+.2f} steer={steer:+.2f} "
                      f"-> L={left:+6.1f} R={right:+6.1f} (speed={speed:.2f})")

            after_drive()   # jog: auto-stop after the burst; held: track release

    except KeyboardInterrupt:
        pass
    finally:
        try:
            chassis.stop()
        except Exception:
            pass
        try:
            chassis.close()
        except Exception:
            pass
        print("\n[stopped, GPIO released]")


if __name__ == "__main__":
    main()
