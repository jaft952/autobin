from __future__ import annotations
from typing import Any
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

class EmergencyStopLayer(BaseLayer):
    """
    Layer 5: Emergency Stop
    Priority: 5 (High)
    Behavior: Suppresses all lower priority layers to halt movement
              if imminent collision or extreme danger is detected.
    """
    def __init__(self):
        super().__init__(layer_id=5)

    def evaluate(self, sensors: Any) -> ActionCommand:
        """
        Polls sensors for obstacle proximity.
        """
        if sensors.has_obstacle():
            # Suppress normal commands, send full stop
            return ActionCommand(
                layer_id=self.layer_id, 
                active=True,
                motion_vector=(0, 0, 0),  # Halt all wheels
                arm_action='stop',        # Halt robot arm
                message="EMERGENCY OBSTACLE STOP"
            )
            
        return ActionCommand(layer_id=self.layer_id, active=False)