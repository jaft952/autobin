import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arm.grasp_planner import GraspPlanner

def print_menu():
    print("="*40)
    print("command :")
    print("  'x y z' : move to specific position in meters (such as: 0.15 0 0.1)")
    print("  'g'     : grasp (close gripper)")
    print("  'r'     : release (open gripper)")
    print("  'h'     : home (return to neutral, arm straight up)")
    print("  'q'     : quit the test program")
    print("-"*40)
    print("  workspace, gripper down (auto-tilts up to 45° when needed):")
    print("    straight down: z=5cm r<~19 | z=10cm r<~18 | z=20cm r<~10")
    print("    with 45° tilt: z=5cm r<~27 | z=10cm r<~27 | z=20cm r<~22")
    print("  x,y,z are SIGNED components in meters (x=right, y=forward, z=up),")
    print("  NOT distances. e.g. '0 0.15 0.1' = 15cm forward, 10cm high.")
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
    
    while True:
        try:
            user_input = input("\n> enter command here: ").strip().lower()
            
            if user_input == 'q':
                print("quitting test program...")
                break
            
            elif user_input == 'h':
                print("back to home position ...")
                planner.home()
                continue
                
            elif user_input == 'g':
                print("executing grasp command ...")
                planner.control_gripper("close")
                continue
                
            elif user_input == 'r':
                print("executing release command ...")
                planner.control_gripper("open")
                continue
                
            parts = user_input.split()
            if len(parts) == 3:
                x, y, z = [float(p) for p in parts]
                print(f"moving to position: X={x}, Y={y}, Z={z} ...")
                
                success = planner.move_to([x, y, z])
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