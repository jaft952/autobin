from abc import ABC, abstractmethod
from src.subsumption.arbitrator import ActionCommand

class BaseLayer(ABC):
    """Base class for subsumption layers."""
    def __init__(self, layer_id: int):
        self.layer_id = layer_id

    @abstractmethod
    def evaluate(self, sensors) -> ActionCommand:
        """Evaluate sensor data, return an active/inactive ActionCommand."""
        return ActionCommand(layer_id=self.layer_id, active=False)

    def notify_arbitration(self, won: bool) -> None:
        """Called each tick with whether this layer won arbitration; timed layers pause on loss."""
        pass

    def reset(self) -> None:
        """Abandon any in-progress manoeuvre, so a resumed layer has no stale timer."""
        pass
