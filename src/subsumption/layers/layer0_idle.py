from typing import Any
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

class SystemIdleLayer(BaseLayer):
    """Layer 0: stops the robot when no other layer is active."""
    def __init__(self):
        super().__init__(layer_id=0)

    def evaluate(self, sensors: Any) -> ActionCommand:
        """Always active, so the robot stops when idle."""
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),
            arm_action='stow',
            message="IDLE"
        )