from abc import ABC, abstractmethod
from src.subsumption.arbitrator import ActionCommand

class BaseLayer(ABC):
    """
    Abstract Base Class for Subsumption Architecture Layers.
    """
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        
    @abstractmethod
    def evaluate(self, sensors) -> ActionCommand:
        """
        Evaluate sensor data and return an active/inactive ActionCommand.
        
        Args:
            sensors: Object matching the SensorInterface that provides getters
                     for polling physical environment data.
        """
        return ActionCommand(layer_id=self.layer_id, active=False)