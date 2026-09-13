from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ActionCommand:
    """A command sent by a layer."""
    layer_id: int
    active: bool
    motion_vector: Optional[tuple] = None
    arm_action: Optional[str] = None
    message: str = ""
    arm_params: Optional[Dict] = None

class Arbitrator:
    """Picks which layer controls the robot."""
    def __init__(self):
        self._current_votes: Dict[int, ActionCommand] = {}

    def submit_command(self, command: ActionCommand):
        """A layer sends its command here."""
        self._current_votes[command.layer_id] = command

    def get_winning_action(self) -> ActionCommand:
        """Return the command of the highest active layer."""
        winning_cmd = ActionCommand(layer_id=-1, active=False, message="Idle")

        for layer_id in sorted(self._current_votes.keys(), reverse=True):
            cmd = self._current_votes[layer_id]
            if cmd.active:
                winning_cmd = cmd
                break

        return winning_cmd

    def clear(self):
        """Clear commands for the next cycle."""
        self._current_votes.clear()
