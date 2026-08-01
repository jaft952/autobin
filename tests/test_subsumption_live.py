"""
tests/test_subsumption_live.py

FULL subsumption stack on the robot — the closest thing to main.py so far:

    SensorHub(camera + ultrasonic)
      -> layers 0 idle / 1 zigzag scan / 2 approach / 3 collect / 5 emergency
      -> Arbitrator
      -> MotionExecutor (wheels) + ArmExecutor (arm)

Expected behavior: the robot zigzag-patrols the floor; when YOLO spots a tin
it arcs toward it; when the tin's ground-contact point enters the calibrated
arc-grasp region (or the IK fallback annulus) the base halts, the arm grabs,
dumps into the onboard bin, homes — and the patrol resumes.

Stage the bring-up with the flags (each stage assumes the previous passed):

    python tests/test_subsumption_live.py --no-motors --no-arm   # decisions only
    python tests/test_subsumption_live.py --no-arm               # driving, no arm
    python tests/test_subsumption_live.py                        # THE WHOLE ROBOT

    --no-camera additionally drops layers 2/3 input (pure zigzag patrol,
    same ground as tests/test_floor_scan.py).

Prerequisites on the Pi:
    - tests/test_ultrasonic.py reads sane distances
    - tests/test_floor_scan.py zigzags correctly (TURN_90_S/SHIFT_S calibrated)
    - arc grid calibrated via tests/test_arc_grasp.py (and/or pixel_to_arm
      calibrated for the IK fallback)

NOTE: a grab is a blocking multi-second sequence — the loop (and printouts)
pause during it. That is by design; the base is halted while it runs.

Ctrl+C stops the motors and releases GPIO.
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.actuators.print_actuator import PrintActuator
from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor
from src.hardware.sensors.sensor_hub import SensorHub
from src.subsumption.arbitrator import Arbitrator
from src.subsumption.motion_executor import MotionExecutor
from src.subsumption.arm_executor import ArmExecutor
from src.subsumption.layers.layer0_idle import SystemIdleLayer
from src.subsumption.layers.layer1_scan import ScanAroundLayer
from src.subsumption.layers.layer2_approach import ApproachLitterLayer
from src.subsumption.layers.layer3_collect import CollectLitterLayer
from src.subsumption.layers.layer5_emergency import EmergencyStopLayer


class PrintPlanner:
    """--no-arm: GraspPlanner stand-in that narrates instead of moving."""

    def goto(self, target, label=""):
        print(f"   [arm] goto({target})")

    def collect(self, pose, tin_pose="upright", dump=True):
        print(f"   [arm] collect({[round(v, 1) for v in pose]}, {tin_pose})")
        return True

    def ik_move(self, xyz, **kw):
        print(f"   [arm] ik_move({[round(v, 3) for v in xyz]})")
        return True

    def open_gripper(self):
        print("   [arm] gripper open")

    def close_gripper(self):
        print("   [arm] gripper close")

    def dump_to_bin(self):
        print("   [arm] dump_to_bin()")


def main():
    ap = argparse.ArgumentParser(description="Full subsumption stack, live")
    ap.add_argument("--no-motors", action="store_true", help="print wheel commands only")
    ap.add_argument("--no-arm", action="store_true", help="print arm actions only")
    ap.add_argument("--no-camera", action="store_true", help="skip YOLO (pure patrol)")
    ap.add_argument("--hz", type=float, default=10.0, help="control loop rate")
    args = ap.parse_args()

    camera = None
    if not args.no_camera:
        # Deferred import: pulls in ultralytics/torch, Pi-only in practice.
        from src.hardware.sensors.camera_sensor import CameraSensor
        camera = CameraSensor()
    sensors = SensorHub(ultrasonic=UltrasonicSensor(), camera=camera)

    layers = [SystemIdleLayer(), ScanAroundLayer(), ApproachLitterLayer(),
              CollectLitterLayer(), EmergencyStopLayer()]
    arbitrator = Arbitrator()
    motion = MotionExecutor(actuator=PrintActuator() if args.no_motors else None)
    arm = ArmExecutor(planner=PrintPlanner() if args.no_arm else None)

    period = 1.0 / args.hz
    print(f"subsumption live @ {args.hz:.0f} Hz — "
          f"motors {'PRINT' if args.no_motors else 'LIVE'}, "
          f"arm {'PRINT' if args.no_arm else 'LIVE'}, "
          f"camera {'OFF' if args.no_camera else 'ON'} — Ctrl+C to stop")

    sensors.start()
    last_msg = None
    try:
        while True:
            tick = time.monotonic()

            sensors.update()
            for layer in layers:
                arbitrator.submit_command(layer.evaluate(sensors))
            winning = arbitrator.get_winning_action()
            motion.execute(winning)
            arm.execute(winning)
            arbitrator.clear()

            msg = f"[L{winning.layer_id}] {winning.message}"
            if msg != last_msg:
                print(msg)
                last_msg = msg

            sleep_left = period - (time.monotonic() - tick)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        motion.close()
        sensors.stop()


if __name__ == "__main__":
    main()
