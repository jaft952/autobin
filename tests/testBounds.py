import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

SAFE_HOME = [90,90,90,90,90,90]
CURRENT_ANGLES = SAFE_HOME.copy()
STOP = False

for i in range(6):
    kit.servo[i].actuation_range = 180
    
def go_home():
    for i in range(6):
        try:
            kit.servo[i].angle = SAFE_HOME[i]
            CURRENT_ANGLES[i] = SAFE_HOME[i]
            time.sleep(0.5)
        except Exception as e:
            print(f"Error occurred while setting servo {i} angle: {e}")
            
def release_all():
    for i in range(6):
        try:
            kit.servo[i].angle = None
        except Exception as e:
            print(f"Error occurred while releasing servo {i}: {e}")

def emergency_stop():
    global STOP
    STOP = True
    release_all()
    print("Emergency stop activated! All servos released.")

def build_angle_sequence(start_angle, end_angle, step):
    step_size = max(1, int(round(abs(step))))
    start_i = int(round(start_angle))
    end_i = int(round(end_angle))
    if start_i <= end_i:
        angles = list(range(start_i, end_i + 1, step_size))
    else:
        angles = list(range(start_i, end_i - 1, -step_size))
    if angles and angles[-1] != end_i:
        angles.append(end_i)
    return angles

def move_channel(channel_id, target_angle, step=2, delay=0.05, prefix=""):
    global STOP
    start_angle = CURRENT_ANGLES[channel_id]
    angles = build_angle_sequence(start_angle, target_angle, step)
    for angle in angles:
        if STOP:
            break
        try:
            label = f"{prefix} " if prefix else ""
            print(f"{label}CH{channel_id+1} -> {angle}")
            kit.servo[channel_id].angle = angle
            CURRENT_ANGLES[channel_id] = angle
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while setting servo {channel_id} angle: {e}")
            return
    
def step_test(channel_id, start_angle, end_angle, step=2, delay=0.05):
    move_channel(channel_id, start_angle, step, delay, prefix="Move to start")
    if STOP:
        return
    move_channel(channel_id, end_angle, step, delay, prefix="Test")

def ask_int_in_range(prompt, min_value, max_value):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if min_value <= value <= max_value:
            return value
        print(f"Please enter a value between {min_value} and {max_value}.")

def ask_float_in_range(prompt, min_value=None, max_value=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if min_value is not None and value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return value

try:
    print("Returning to safe home position...")
    go_home()
    time.sleep(2)

    while not STOP:
        ch = ask_int_in_range("channel (1-6): ", 1, 6) - 1
        start = ask_float_in_range("start angle (0-180): ", 0, 180)
        end = ask_float_in_range("end angle (0-180): ", 0, 180)
        step = ask_float_in_range("step (>0): ", 0.0001)
        delay = ask_float_in_range("delay (s, >=0): ", 0)

        step_test(ch, start, end, step, delay)
        if STOP:
            break

        cmd = input("Continue? (y/n, e=emergency stop): ").strip().lower()
        if cmd == 'e':
            emergency_stop()
            break
        if cmd not in ('y', 'yes'):
            print("Quitting test...")
            break
except KeyboardInterrupt:
    emergency_stop()
finally: 
    try: 
        release_all()
    except Exception as e:
        print(f"Error occurred while releasing all servos: {e}")