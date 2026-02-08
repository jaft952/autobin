#!/usr/bin/env python3
from adafruit_servokit import ServoKit
import time

# 初始化PCA9685
kit = ServoKit(channels=16)

def move_servo(channel, angle):
    try:
        kit.servo[channel].angle = angle
        return True
    except Exception as e:
        print(f"error : {e}")
        return False

def test_channel(channel):
    print(f"\n{'='*50}")
    print(f"test channel: {channel}")
    print(f"{'='*50}")
    
    print("move to 90° (middle)...")
    move_servo(channel, 90)
    time.sleep(1)
    
    print("start swing test...")
    for _ in range(3):
        print("  → 0°")
        move_servo(channel, 0)
        time.sleep(0.8)
        
        print("  → 180°")
        move_servo(channel, 180)
        time.sleep(0.8)
    
    print("  → 90° (complete)")
    move_servo(channel, 90)
    time.sleep(0.5)

def interactive_mode():
    """Interactive mode"""
    print("\n" + "="*60)
    print("         6 DOF Robotic Arm Servo Identifier")
    print("="*60)
    print("\nInstructions:")
    print("  • Enter 1-6: Test corresponding channel (Channels 0-5)")
    print("  • Enter 'a' or 'all': Test all channels sequentially")
    print("  • Enter 'q' or 'quit': Quit")
    print("  • Enter 'r' or 'reset': Reset all servos to 90°")
    print("="*60)
    
    while True:
        print("\nEnter command: ", end="")
        user_input = input().strip().lower()
        
        # Quit
        if user_input in ['q', 'quit', 'exit']:
            print("\nExiting program...")
            break
        
        # Reset all servos
        elif user_input in ['r', 'reset']:
            print("\nResetting all servos to 90°...")
            for i in range(6):
                kit.servo[i].angle = 90
                time.sleep(0.2)
            print("✓ Reset complete")
        
        # Test all channels
        elif user_input in ['a', 'all']:
            print("\nStarting test for all channels...")
            for i in range(6):
                print(f"Testing channel {i+1} (Channel {i})")
                test_channel(i)
                
                if i < 5:  # Not the last one
                    print("\nPress Enter to continue to the next, or enter 's' to skip: ", end="")
                    skip = input().strip().lower()
                    if skip == 's':
                        print("Skipping remaining tests")
                        break
            
            print("\n✓ All tests completed")
        
        # Test single channel
        elif user_input.isdigit():
            channel_num = int(user_input)
            if 1 <= channel_num <= 6:
                channel = channel_num - 1  
                test_channel(channel)
                
                # Ask which part this is
                print(f"\nWhich part is channel {channel_num}?")
                print("Enter part name or press Enter to skip: ", end="")
                part_name = input().strip()
                
                if part_name:
                    print(f"✓ Recorded: Channel {channel_num} = {part_name}")
                    # Optionally save to file
                    with open("servo_mapping.txt", "a") as f:
                        f.write(f"Channel {channel}: {part_name}\n")
                    print("  (Saved to servo_mapping.txt)")
            else:
                print("✗ Error: Please enter a number between 1 and 6")
        
        # Invalid input
        else:
            print("✗ Invalid command, please try again")

def main():
    """Main program"""
    try:
        # Enter interactive mode
        interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n✗ Program interrupted")
        print("Resetting all servos...")
        for i in range(6):
            kit.servo[i].angle = 90
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
