from typing import Any
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

class SystemIdleLayer(BaseLayer):
    """
    Layer 0: System Idle
    Priority: 0 (Lowest)
    Behavior: The absolute default fallback layer if no other layer is active.
              Simply maintains zero forward momentum.
    """
    def __init__(self):
        super().__init__(layer_id=0)

    def evaluate(self, sensors: Any) -> ActionCommand:
        """
        Always returns active to ensure the robot stops when idle.
        """
        return ActionCommand(
            layer_id=self.layer_id, 
            active=True,
            motion_vector=(0, 0, 0),  # Idle state, stop the base
            arm_action='stow',        # Retract/stow arm when inactive
            message="IDLE"
        )