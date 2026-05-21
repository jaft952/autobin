from typing import Any
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

class CollectLitterLayer(BaseLayer):
    """
    Layer 3: Collect Litter
    Priority: 3 (Medium)
    Behavior: Executes the robotic arm sequence for fetching ground
              litter when in close proximity.
    """
    def __init__(self):
        super().__init__(layer_id=3)

    def evaluate(self, sensors: Any) -> ActionCommand:
        """
        Polls sensors. If Target is within arm reach -> Stop and Grab.
        """
        litter_pos = sensors.get_litter_position()
        if litter_pos:
            x, y = litter_pos
            # Simple threshold check for arm reachability
            if abs(x) < 0.2 and y < 0.3:
                return ActionCommand(
                    layer_id=self.layer_id, 
                    active=True,
                    motion_vector=(0, 0, 0),  # Halt base
                    arm_action='grab_sequence',# Start grab state machine execution in actuator
                    message="EXECUTING LITTER COLLECTION SEQUENCE"
                )
                
        return ActionCommand(layer_id=self.layer_id, active=False)