#!/usr/bin/env python3
from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

NEUTRAL = 135  # All servos neutral at 135°

def move_servo(channel, angle):
    try:
        kit.servo[channel].angle = angle
        return True
    except Exception as e:
        print(f"error : {e}")
        return False

def reset_all():
    print("Resetting all servos to neutral (135°)...")
    for i in range(6):
        kit.servo[i].angle = NEUTRAL
        time.sleep(0.1)
    print("✓ Reset complete")

def test_channel(channel):
    print(f"\n{'='*50}")
    print(f"test channel: {channel}")
    print(f"{'='*50}")

    print("move to 135° (neutral)...")
    move_servo(channel, NEUTRAL)
    time.sleep(1)

    print("start swing test...")
    for _ in range(3):
        print("  → 0°")
        move_servo(channel, 0)
        time.sleep(0.8)

        print("  → 270°")
        move_servo(channel, 270)
        time.sleep(0.8)

    print("  → 135° (complete)")
    move_servo(channel, NEUTRAL)
    time.sleep(0.5)

def direction_test():
    """
    Direction test: move one servo at a time to 90° (neutral-45°)
    and observe which direction the arm moves.
    """
    print("\n" + "="*60)
    print("         Direction Test Mode")
    print("="*60)
    print("Purpose: Find out which direction each servo rotates")
    print("All servos will be set to neutral (135°) first.")
    print("Then ONE servo will move to 90° (45° less than neutral).")
    print("Observe the direction and record it.")
    print("="*60)

    channels = {
        1: "CH1 - Base Rotation (should turn LEFT or RIGHT?)",
        2: "CH2 - Shoulder      (should bend FORWARD or BACKWARD?)",
        3: "CH3 - Elbow         (should bend FORWARD or BACKWARD?)",
        4: "CH4 - Wrist Pitch   (should bend UP or DOWN?)",
        5: "CH5 - Wrist Rotate  (should rotate CW or CCW?)",
    }

    while True:
        print("\nWhich channel to test?")
        for k, v in channels.items():
            print(f"  {k}: {v}")
        print("  r: Reset all to neutral")
        print("  b: Back to main menu")
        print("Enter: ", end="")

        user_input = input().strip().lower()

        if user_input == 'b':
            break

        elif user_input == 'r':
            reset_all()

        elif user_input.isdigit() and 1 <= int(user_input) <= 5:
            ch = int(user_input) - 1  # 0-indexed
            ch_name = channels[int(user_input)]

            print(f"\nTesting: {ch_name}")
            print("Step 1: Reset all to neutral (135°)...")
            reset_all()
            time.sleep(1)

            print(f"Step 2: Moving CH{int(user_input)} to 90° (neutral - 45°)...")
            print(">>> OBSERVE THE ARM NOW <<<")
            move_servo(ch, 90)
            time.sleep(2)

            print(f"\nWhich direction did it move?")
            print("Enter your observation (e.g. 'left', 'right', 'forward', 'backward'): ", end="")
            obs = input().strip()

            print(f"\nStep 3: Moving CH{int(user_input)} to 180° (neutral + 45°)...")
            print(">>> OBSERVE THE ARM NOW <<<")
            move_servo(ch, 180)
            time.sleep(2)

            print(f"\nWhich direction did it move now?")
            print("Enter your observation: ", end="")
            obs2 = input().strip()

            print(f"\n✓ Recorded for CH{int(user_input)}:")
            print(f"   90°  (neutral-45°) → {obs}")
            print(f"   180° (neutral+45°) → {obs2}")

            with open("direction_mapping.txt", "a") as f:
                f.write(f"CH{int(user_input)} ({ch_name}):\n")
                f.write(f"  90°  → {obs}\n")
                f.write(f"  180° → {obs2}\n\n")
            print("  (Saved to direction_mapping.txt)")

            print("\nResetting back to neutral...")
            reset_all()

        else:
            print("✗ Invalid input")


def interactive_mode():
    print("\n" + "="*60)
    print("         6 DOF Robotic Arm Servo Identifier")
    print("="*60)
    print("\nInstructions:")
    print("  • Enter 1-6: Test corresponding channel (Channels 0-5)")
    print("  • Enter 'd': Direction test mode (find axis directions)")
    print("  • Enter 'a' or 'all': Test all channels sequentially")
    print("  • Enter 'q' or 'quit': Quit")
    print("  • Enter 'r' or 'reset': Reset all servos to neutral (135°)")
    print("="*60)

    while True:
        print("\nEnter command: ", end="")
        user_input = input().strip().lower()

        if user_input in ['q', 'quit', 'exit']:
            print("\nExiting program...")
            break

        elif user_input in ['r', 'reset']:
            reset_all()

        elif user_input == 'd':
            direction_test()

        elif user_input in ['a', 'all']:
            print("\nStarting test for all channels...")
            for i in range(6):
                print(f"Testing channel {i+1} (Channel {i})")
                test_channel(i)
                if i < 5:
                    print("\nPress Enter to continue, or 's' to skip: ", end="")
                    skip = input().strip().lower()
                    if skip == 's':
                        break
            print("\n✓ All tests completed")

        elif user_input.isdigit():
            channel_num = int(user_input)
            if 1 <= channel_num <= 6:
                channel = channel_num - 1
                test_channel(channel)
                print(f"\nWhich part is channel {channel_num}?")
                print("Enter part name or press Enter to skip: ", end="")
                part_name = input().strip()
                if part_name:
                    print(f"✓ Recorded: Channel {channel_num} = {part_name}")
                    with open("servo_mapping.txt", "a") as f:
                        f.write(f"Channel {channel}: {part_name}\n")
                    print("  (Saved to servo_mapping.txt)")
            else:
                print("✗ Error: Please enter a number between 1 and 6")

        else:
            print("✗ Invalid command, please try again")


def main():
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\n✗ Program interrupted")
        print("Resetting all servos...")
        reset_all()
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()