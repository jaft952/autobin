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
        
    def get_litter_position(self) -> tuple:
        """Returns (x, y) coordinates of ground litter relative to robot, or None"""
        return None
        
    def get_aerial_trash_position(self) -> tuple:
        """Returns (x, y, z) coordinates of airborne trash, or None"""
        return None