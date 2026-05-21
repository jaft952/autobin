from typing import Any
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

class ScanAroundLayer(BaseLayer):
    """
    Layer 1: Scan Around
    Priority: 1 (Very Low)
    Behavior: Enables slow rotation to search for targets using cameras/sensors.
              Overrides idle to allow autonomous patrolling.
    """
    def __init__(self):
        super().__init__(layer_id=1)

    def evaluate(self, sensors: Any) -> ActionCommand:
        """
        Polls sensors. If no target found, rotates.
        """
        if not sensors.get_litter_position() and not sensors.get_aerial_trash_position():
            return ActionCommand(
                layer_id=self.layer_id, 
                active=True,
                motion_vector=(0, 0, 0.5),  # Pure rotation
                arm_action='stow',          # Keep arm stowed
                message="SCANNING ENVIRONMENT"
            )
            
        return ActionCommand(layer_id=self.layer_id, active=False)