import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.visual_servoing.closed_loop_servo import ClosedLoopServo, ServoState

def test_simulation():

    servo = ClosedLoopServo()
    
    servo.simulation_mode = True 
    
    fake_target_tin = (320, 200) 
    
    print("=== start simulation ===")
    print(f"target set at (320, 150)")
    print(f"initial target position: {fake_target_tin}")
    
    for frame_id in range(1, 12):
        print(f"\n[Frame {frame_id}]")
        
        current_state, (dx, dy) = servo.update(frame=None, target_pos=fake_target_tin)
        
        print(f"current State  : {current_state.name} (Enum Value: {current_state.value})")
        print(f"calculated offset : dx={dx:.2f} mm, dy={dy:.2f} mm")
        
        new_ty = fake_target_tin[1] - 5
        fake_target_tin = (320, new_ty)
        
        if current_state == ServoState.GRASPING:
            print("\n✅ Target reached and grasped successfully!")
            break
            
        time.sleep(0.5)

if __name__ == "__main__":
    test_simulation()