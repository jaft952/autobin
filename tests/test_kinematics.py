import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.grasp_planner import GraspPlanner
from src.arm.kinematics import GRIPPER_DOWN

def print_menu():
    print("="*40)
    print("command :")
    print("  'x y z' : move to specific position in meters (such as: 0.15 0 0.1)")
    print("  'down'  : gripper-DOWN mode (constrained, for grasping)")
    print("  'free'  : POSITION-ONLY mode (no orientation, comfortable poses)")
    print("  'g'     : grasp (close gripper)")
    print("  'r'     : release (open gripper)")
    print("  'h'     : home ")
    print("  'bin'   : throw-to-bin pose")
    print("  'sweet' : sweet point 1 grasp pose")
    print("  'q'     : quit the test program")
    print("-"*40)
    print("  x,y,z are SIGNED components in meters (x=right, y=forward, z=up),")
    print("  NOT distances. e.g. '0 0.15 0.1' = 15cm forward, 10cm high.")
    print("  TIP: high z + small horizontal = low torque (less sag).")
    print("="*40)

def main():
    try:
        planner = GraspPlanner()
        print("successfully initialized GraspPlanner. Ready for commands.")
    except Exception as e:
        print(f"failed to initialize GraspPlanner: {e}")
        print("Please check your robot's power and I2C connection (PCA9685).")
        return

    print_menu()
    mode = {"tool": GRIPPER_DOWN, "name": "DOWN"}  # default: gripper points down

    while True:
        try:
            user_input = input(f"\n[{mode['name']}] enter command here: ").strip().lower()

            if user_input == 'q':
                print("quitting test program...")
                break

            elif user_input == 'h':
                print("back to home position ...")
                planner.home()
                continue

            elif user_input == 'down':
                mode["tool"], mode["name"] = GRIPPER_DOWN, "DOWN"
                print("mode: gripper DOWN (constrained)")
                continue

            elif user_input == 'free':
                mode["tool"], mode["name"] = None, "FREE"
                print("mode: POSITION-ONLY (no orientation constraint)")
                continue

            elif user_input == 'g':
                print("executing grasp command ...")
                planner.control_gripper("close")
                continue

            elif user_input == 'r':
                print("executing release command ...")
                planner.control_gripper("open")
                continue

            elif user_input == 'bin':
                print("moving to throw-to-bin pose ...")
                planner.goto_named_pose("bin")
                continue

            elif user_input in ('sweet', 's1', 'sweet1'):
                print("moving to sweet point 1 ...")
                planner.goto_named_pose("sweet1")
                continue

            parts = user_input.split()
            if len(parts) == 3:
                x, y, z = [float(p) for p in parts]
                print(f"moving to position: X={x}, Y={y}, Z={z} ({mode['name']}) ...")

                success = planner.move_to([x, y, z], tool_direction=mode["tool"])
                if success:
                    print(f"✅ successfully reached target position! ({x}, {y}, {z})")
                else:
                    print("⚠️ cannot reach target position. Please check if it's within the robot's workspace and try again.")
            else:
                print("Invalid command. Please enter 'x y z' or select a single-letter command from the menu.")
                
        except ValueError:
            print("Invalid input format! If you want to enter coordinates, make sure to include three numbers separated by spaces. For example: '10 -5 15'")
        except KeyboardInterrupt:
            print("\nProgram terminated by user. Shutting down...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()