from typing import Any
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

class InterceptTrashLayer(BaseLayer):
    """
    Layer 4: Intercept Airborne Trash
    Priority: 4 (Medium-High)
    Behavior: Triggers the rapid aerial interception module (Module 1).
    """
    def __init__(self):
        super().__init__(layer_id=4)

    def evaluate(self, sensors: Any) -> ActionCommand:
        """
        Polls sensors. If aerial target exists, take over immediately.
        """
        aerial_pos = sensors.get_aerial_trash_position()
        if aerial_pos:
            return ActionCommand(
                layer_id=self.layer_id, 
                active=True,
                motion_vector=(0, 0, 0),  # Typically base is static for airborne
                arm_action='intercept_fast',
                message="INTERCEPTING AIRBORNE TARGET"
            )
            
        return ActionCommand(layer_id=self.layer_id, active=False)