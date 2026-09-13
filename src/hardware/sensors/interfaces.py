class SensorInterface:
    """Base class for all sensors."""
    def update(self):
        """Read the sensor."""
        pass

    def has_obstacle(self) -> bool:
        """True if something is about to be hit."""
        return False

    def get_obstacle_distance_cm(self):
        """Distance to the nearest object in front, or None."""
        return None

    def get_obstacle_distance_back_cm(self):
        """Distance to the nearest object behind, or None."""
        return None

    def get_obstacle_distance_front_left_cm(self):
        """Distance from the front-left sensor, or None."""
        return None

    def get_obstacle_distance_front_right_cm(self):
        """Distance from the front-right sensor, or None."""
        return None

    def get_litter_position(self) -> tuple:
        """Position of the litter, or None."""
        return None

    def get_litter_locked(self) -> bool:
        """True while a can is locked."""
        return False

    def get_litter_ground_contact(self) -> tuple:
        """Point where the litter touches the floor, or None."""
        return None

    def get_litter_distance_cm(self):
        """Estimated distance to the locked litter, or None."""
        return None

    def get_litter_too_close(self) -> bool:
        """True if the litter is too close and the robot should back off."""
        return False

    def get_litter_target_error(self):
        """Steering error to the locked litter, or None."""
        return None

    def get_litter_pose(self) -> dict:
        """Pose of the litter from its mask, or None."""
        return None

    def get_aerial_trash_position(self) -> tuple:
        """Position of trash in the air, or None."""
        return None