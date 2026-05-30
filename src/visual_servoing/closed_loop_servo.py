import enum
import math
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

class ServoState(enum.Enum):
    IDLE = 0
    SEARCHING = 1
    APPROACHING = 2
    GRASPING = 3
    DONE = 4

class ClosedLoopServo:
    """
    Closed loop visual servoing controller.
    Uses pixel error to calculate movement commands for the robotic arm.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.state = ServoState.IDLE
        self.simulation_mode = True
        
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "servo_config.yaml"
            
        self.config = self._load_config(config_path)
        
        # Load parameters
        self.pixel_to_mm = self.config.get("pixel_to_mm", 1.0)
        self.gain = self.config.get("gain", 0.2)
        self.error_threshold_px = self.config.get("error_threshold_px", 10.0)
        self.max_step_mm = self.config.get("max_step_mm", 5.0)
        
        self.gripper_detector = None
        
    def _load_config(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def update(self, frame: Any, target_pos: Optional[Tuple[int, int]]) -> Tuple[ServoState, Tuple[float, float]]:
        """
        Calculates (dx, dy) movement based on the current state.
        Returns Current State and a tuple of (dx_mm, dy_mm).
        """
        if self.state == ServoState.IDLE:
            if target_pos is not None:
                self.state = ServoState.APPROACHING
            return self.state, (0.0, 0.0)
            
        if self.state == ServoState.APPROACHING:
            if target_pos is None:
                self.state = ServoState.SEARCHING
                return self.state, (0.0, 0.0)
                
            tx, ty = target_pos
            
            # Step 1: Define Feature Points
            if self.simulation_mode:
                gx, gy = 320, 150
            else:
                if self.gripper_detector is None:
                    return self.state, (0.0, 0.0)
                
                detected = self.gripper_detector.detect(frame)
                if detected is None:
                    return self.state, (0.0, 0.0)
                gx, gy, angle_deg, open_width = detected
                
            # Pixel error
            ex = tx - gx
            ey = ty - gy
            error_norm = math.hypot(ex, ey)
            
            # Step 3: Convergence condition
            if error_norm < self.error_threshold_px:
                self.state = ServoState.GRASPING
                return self.state, (0.0, 0.0)
                
            # Step 2: Calculate Control Velocity (dx, dy)
            dx_mm = ex * self.pixel_to_mm * self.gain
            dy_mm = ey * self.pixel_to_mm * self.gain
            
            # Step 4: Safety constraints
            step_norm = math.hypot(dx_mm, dy_mm)
            if step_norm > self.max_step_mm:
                scale = self.max_step_mm / step_norm
                dx_mm *= scale
                dy_mm *= scale
                
            return self.state, (dx_mm, dy_mm)
            
        if self.state == ServoState.SEARCHING:
            if target_pos is not None:
                self.state = ServoState.APPROACHING
            return self.state, (0.0, 0.0)
            
        if self.state == ServoState.GRASPING:
            # Trigger grasp routine here
            self.state = ServoState.DONE
            return self.state, (0.0, 0.0)
            
        return self.state, (0.0, 0.0)
