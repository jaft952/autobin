from abc import ABC, abstractmethod
from src.subsumption.arbitrator import ActionCommand

class BaseLayer(ABC):
    """Base class for all layers."""
    def __init__(self, layer_id: int):
        self.layer_id = layer_id

    @abstractmethod
    def evaluate(self, sensors) -> ActionCommand:
        """Read the sensors and return this layer's command."""
        return ActionCommand(layer_id=self.layer_id, active=False)

    def notify_arbitration(self, won: bool) -> None:
        """Told every tick whether this layer won."""
        pass

    def reset(self) -> None:
        """Stop any move in progress."""
        pass
