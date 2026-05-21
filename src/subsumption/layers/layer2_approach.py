from typing import Any
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

class ApproachLitterLayer(BaseLayer):
    """
    Layer 2: Approach Litter
    Priority: 2 (Low)
    Behavior: Drives the base towards detected ground litter.
    """
    def __init__(self):
        super().__init__(layer_id=2)

    def evaluate(self, sensors: Any) -> ActionCommand:
        """
        Calculates simple alignment vectors.
        """
        litter_pos = sensors.get_litter_position()
        if litter_pos:
            x, y = litter_pos
            # Simplified pseudo-code approach 
            # In a real system, use PID or pure pursuit based on x,y
            forward_speed = 1.0 if y > 0.5 else 0.2
            steer = x * 0.5 
            
            return ActionCommand(
                layer_id=self.layer_id, 
                active=True,
                motion_vector=(forward_speed, 0, steer),  
                arm_action='deploy',        
                message="APPROACHING GROUND LITTER"
            )
            
        return ActionCommand(layer_id=self.layer_id, active=False)