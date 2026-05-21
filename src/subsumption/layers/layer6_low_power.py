from typing import Any
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

class LowPowerModeLayer(BaseLayer):
    """
    Layer 6: Low Power Mode
    Priority: 6 (Highest)
    Behavior: Forces safe shutdown / go-home routine if critical battery
              is detected, overriding all navigation.
    """
    def __init__(self):
        super().__init__(layer_id=6)

    def evaluate(self, sensors: Any) -> ActionCommand:
        """
        Check battery status from sensors.
        """
        if sensors.get_battery_level() < 0.1:
            return ActionCommand(
                layer_id=self.layer_id, 
                active=True,
                motion_vector=(0, 0, 0),  # Or navigate home logic
                arm_action='stow',        
                message="CRITICAL BATTERY: FORCING LOW POWER SHUTDOWN"
            )
            
        return ActionCommand(layer_id=self.layer_id, active=False)