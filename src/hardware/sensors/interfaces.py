class SensorInterface:
    """Base class for Hardware Sensors using Polled Abstraction."""
    def update(self):
        """Poll the hardware to update current state. Called every tick by main loop."""
        pass
        
    def get_battery_level(self) -> float:
        """Returns battery percentage 0.0 - 1.0"""
        return 1.0
        
    def has_obstacle(self) -> bool:
        """Returns True if there is an imminent collision"""
        return False

    def get_obstacle_distance_cm(self):
        """Returns distance (cm) to the nearest forward obstacle, or None if
        unknown/out of range. None must be treated as 'no information',
        never as an obstacle at 0 cm."""
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

    def get_litter_ground_contact(self) -> tuple:
        """Returns normalized (x, y) of the litter's ground-contact point
        (where it touches the floor), or None. Grasp solvers are calibrated
        against this point, not the bounding-box center."""
        return None

    def get_litter_pose(self) -> dict:
        """Returns the litter's pose from segmentation, or None if unknown:
        {'klass': 'upright'|'lying'|'axial', 'angle': deg 0..180}.
        angle is the image-plane long-axis angle (90 = vertical in image);
        it is only meaningful for 'lying'. None must be treated as
        'upright' (the historical assumption)."""
        return None
        
    def get_aerial_trash_position(self) -> tuple:
        """Returns (x, y, z) coordinates of airborne trash, or None"""
        return None