#!/usr/bin/env python3
"""
6 DOF Robotic Arm Real-time Controller
Real-time control like a game controller - hold key to move
"""
from adafruit_servokit import ServoKit
import time
import os
import sys

# Windows specific key detection
if sys.platform == 'win32':
    import msvcrt
else:
    # For Linux/Mac, you might need a different approach
    import sys
    import termios
    import tty

class ServoController:
    def __init__(self):
        self.kit = ServoKit(channels=16)
        self.num_servos = 6
        
        # Current angles for each servo (default to 90°)
        self.angles = [90] * self.num_servos
        
        # Currently selected servo (0-5)
        self.selected_servo = 0
        
        # Movement settings
        self.speed = 2  # Degrees per update
        self.update_interval = 0.05  # 50ms update interval
        
        # Movement flags (for continuous movement)
        self.moving_up = False
        self.moving_down = False
        
        # Servo names (can be customized)
        self.servo_names = [
            "Base",
            "Shoulder", 
            "Elbow",
            "Wrist Pitch",
            "Wrist Roll",
            "Gripper"
        ]
        
        # Key tracking for continuous movement
        self.key_states = {
            '1': False, '2': False, '3': False, '4': False, '5': False, '6': False,
            'up': False, 'down': False, 'w': False, 's': False
        }
    
    def reset_all(self):
        """Reset all servos to 90°"""
        for i in range(self.num_servos):
            self.angles[i] = 90
            self.kit.servo[i].angle = 90
            time.sleep(0.1)
    
    def move_servo(self, channel, angle):
        """Move servo to specified angle with bounds checking"""
        # Clamp angle between 0 and 180
        angle = max(0, min(180, angle))
        
        try:
            self.kit.servo[channel].angle = angle
            self.angles[channel] = angle
            return True
        except Exception as e:
            print(f"\n[ERROR] Channel {channel}: {e}")
            return False
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_ui(self):
        """Display the control interface"""
        self.clear_screen()
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "6 DOF ROBOTIC ARM CONTROLLER" + " " * 30 + "║")
        print("╠" + "═" * 78 + "╣")
        print("║                                                                              ║")
        
        # Display all servos with their current angles
        for i in range(self.num_servos):
            selected = "►" if i == self.selected_servo else " "
            bar = self.create_progress_bar(self.angles[i])
            
            name = f"[{i+1}] {self.servo_names[i]}".ljust(18)
            angle_str = f"{self.angles[i]:6.1f}°"
            
            print(f"║  {selected} {name} {angle_str}  {bar} ║")
        
        print("║                                                                              ║")
        print("╠" + "═" * 78 + "╣")
        print("║  CONTROLS:                                                                   ║")
        print("║  • [1-6]        : Select servo                                               ║")
        print("║  • [↑] / [W]    : Increase angle (hold for continuous movement)              ║")
        print("║  • [↓] / [S]    : Decrease angle (hold for continuous movement)              ║")
        print("║  • [+] / [-]    : Adjust movement speed                                      ║")
        print("║  • [R]          : Reset all servos to 90°                                    ║")
        print("║  • [H]          : Show help menu                                             ║")
        print("║  • [ESC] / [Q]  : Quit                                                       ║")
        print("╠" + "═" * 78 + "╣")
        print(f"║  Speed: {self.speed:.1f}°/tick  |  Selected: [{self.selected_servo+1}] {self.servo_names[self.selected_servo]}" + " " * 28 + "║")
        print("╚" + "═" * 78 + "╝")
    
    def create_progress_bar(self, angle, width=30):
        """Create a visual progress bar for angle (0-180°)"""
        percentage = angle / 180
        filled = int(percentage * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def show_help(self):
        """Show detailed help menu"""
        self.clear_screen()
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 32 + "HELP MENU" + " " * 37 + "║")
        print("╠" + "═" * 78 + "╣")
        print("║                                                                              ║")
        print("║  REAL-TIME CONTROLS (like a game controller):                               ║")
        print("║  ─────────────────────────────────────────────────────────────────────────   ║")
        print("║                                                                              ║")
        print("║  [1-6]          : Press to select servo (hold any key to keep moving)        ║")
        print("║  [↑] / [W]      : Hold to increase angle (release to stop)                   ║")
        print("║  [↓] / [S]      : Hold to decrease angle (release to stop)                   ║")
        print("║                                                                              ║")
        print("║  SPEED CONTROL:                                                              ║")
        print("║  ─────────────────────────────────────────────────────────────────────────   ║")
        print("║  [+]            : Increase speed (max 10°/tick)                              ║")
        print("║  [-]            : Decrease speed (min 0.5°/tick)                             ║")
        print("║                                                                              ║")
        print("║  OTHER COMMANDS:                                                             ║")
        print("║  ─────────────────────────────────────────────────────────────────────────   ║")
        print("║  [R]            : Reset all servos to 90°                                    ║")
        print("║  [H]            : Show this help menu                                        ║")
        print("║  [ESC] / [Q]    : Quit the program                                           ║")
        print("║                                                                              ║")
        print("╚" + "═" * 78 + "╝")
        input("\nPress Enter to return to controller...")
    
    def update_movement(self):
        """Update servo position based on movement flags"""
        if self.moving_up:
            new_angle = self.angles[self.selected_servo] + self.speed
            self.move_servo(self.selected_servo, new_angle)
        elif self.moving_down:
            new_angle = self.angles[self.selected_servo] - self.speed
            self.move_servo(self.selected_servo, new_angle)
    
    def select_servo(self, index):
        """Select servo by index"""
        if 0 <= index < self.num_servos:
            self.selected_servo = index
            return True
        return False
    
    def move_up(self, steps=1):
        """Move selected servo up"""
        new_angle = self.angles[self.selected_servo] + (self.speed * steps)
        self.move_servo(self.selected_servo, new_angle)
    
    def move_down(self, steps=1):
        """Move selected servo down"""
        new_angle = self.angles[self.selected_servo] - (self.speed * steps)
        self.move_servo(self.selected_servo, new_angle)
    
    def set_angle(self, angle):
        """Set selected servo to specific angle"""
        try:
            angle = float(angle)
            self.move_servo(self.selected_servo, angle)
            return True
        except ValueError:
            return False
    
    def increase_speed(self):
        """Increase movement speed"""
        self.speed = min(10, self.speed + 0.5)
    
    def decrease_speed(self):
        """Decrease movement speed"""
        self.speed = max(0.5, self.speed - 0.5)
    
    def get_key(self):
        """Get pressed key without blocking on Windows"""
        if sys.platform == 'win32':
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # Handle special keys (arrow keys return 2 bytes)
                if key == b'\xe0':  # Special key prefix
                    special_key = msvcrt.getch()
                    if special_key == b'H':  # Up arrow
                        return 'up'
                    elif special_key == b'P':  # Down arrow
                        return 'down'
                    return None
                else:
                    try:
                        return key.decode('utf-8').lower()
                    except:
                        return None
        return None
    
    def run(self):
        """Main control loop"""
        print("Initializing controller...")
        time.sleep(1)
        
        self.display_ui()
        
        print("\n[INFO] Controller ready! Press keys to control servos...")
        print("[INFO] Press ESC or Q to quit\n")
        
        try:
            last_display_update = time.time()
            
            while True:
                current_time = time.time()
                
                # Get key press (non-blocking)
                key = self.get_key()
                
                if key:
                    # Number keys (1-6) to select servo
                    if key in ['1', '2', '3', '4', '5', '6']:
                        self.select_servo(int(key) - 1)
                        self.display_ui()
                    
                    # Movement keys
                    elif key in ['up', 'w']:
                        self.moving_up = True
                    elif key in ['down', 's']:
                        self.moving_down = True
                    
                    # Speed adjustment
                    elif key == '+' or key == '=':
                        self.increase_speed()
                        self.display_ui()
                    elif key == '-':
                        self.decrease_speed()
                        self.display_ui()
                    
                    # Reset
                    elif key == 'r':
                        self.reset_all()
                        self.display_ui()
                    
                    # Help
                    elif key == 'h':
                        self.show_help()
                        self.display_ui()
                    
                    # Quit
                    elif key == 'q' or key == '\x1b':  # ESC key
                        break
                
                # Update movement continuously
                self.update_movement()
                
                # Refresh display periodically when moving
                if (self.moving_up or self.moving_down) and (current_time - last_display_update > 0.1):
                    self.display_ui()
                    last_display_update = current_time
                
                # Small delay to prevent CPU spinning
                time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            print("\n\n[INFO] Controller interrupted")
        
        finally:
            print("\n[INFO] Resetting servos...")
            self.reset_all()
            print("[INFO] Controller stopped")

def main():
    """Main entry point"""
    try:
        controller = ServoController()
        controller.run()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()