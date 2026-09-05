from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LitterSnapshot:
    """One frame's locked-tin facts. klass=None means unknown, not upright."""
    center: Optional[tuple]
    ground_contact: Optional[tuple]
    klass: Optional[str]
    angle: Optional[float]


class SensorInterface:
    """Base class for Hardware Sensors using Polled Abstraction."""
    def update(self):
        """Poll the hardware to update current state. Called every tick by main loop."""
        pass

    def has_obstacle(self) -> bool:
        """Returns True if there is an imminent collision"""
        return False

    def get_obstacle_distance_cm(self):
        """Distance (cm) to the nearest forward obstacle, or None if unknown."""
        return None

    def get_obstacle_distance_back_cm(self):
        """Same contract as get_obstacle_distance_cm(), rear-facing sensor."""
        return None

    def get_obstacle_distance_front_left_cm(self):
        """Same contract as get_obstacle_distance_cm(), front-left diagonal sensor."""
        return None

    def get_obstacle_distance_front_right_cm(self):
        """Same contract as get_obstacle_distance_cm(), front-right diagonal sensor."""
        return None

    def get_litter_position(self) -> tuple:
        """Returns (x, y) coordinates of ground litter relative to robot, or None"""
        return None

    def get_litter_locked(self) -> bool:
        """True while a tin is held by TargetLock, even during occlusion grace."""
        return False

    def get_litter_ground_contact(self) -> tuple:
        """Normalized (x, y) ground-contact point of the litter, or None. Grasp solvers use this, not the bbox center."""
        return None

    def snapshot(self):
        """A LitterSnapshot from ONE frame; use this when several litter facts are needed together."""
        return None

    def get_litter_distance_cm(self):
        """Monocular bbox-based distance estimate, or None."""
        return None

    def get_litter_too_close(self) -> bool:
        """True when the locked litter's bbox is close enough to back off regardless of distance estimate."""
        return False

    def get_litter_target_error(self):
        """TargetError for the locked target, via compute_target_error(); None if unavailable."""
        return None

    def get_litter_pose(self) -> dict:
        """Litter pose from segmentation, or None (treat as upright): {'klass', 'angle' (image-plane, lying only)}."""
        return None

    def get_aerial_trash_position(self) -> tuple:
        """Returns (x, y, z) coordinates of airborne trash, or None"""
        return None
