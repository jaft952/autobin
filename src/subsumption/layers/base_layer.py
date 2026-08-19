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

    def notify_arbitration(self, won: bool) -> None:
        """Called once per tick with whether this layer's command was the one
        executed. Layers timing an open-loop manoeuvre must pause their timers
        when suppressed, or they run the sequence while the robot is standing
        still. Default: layers with no timers ignore it."""
        pass