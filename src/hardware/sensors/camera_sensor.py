"""
src/hardware/sensors/camera_sensor.py

CameraSensor — Concrete implementation of SensorInterface.
Bridges AluminiumCanDetector (perception) with the Subsumption layers.

Each tick, update() is called by the main loop.
Layers call get_litter_position() to query detection results.
"""

from src.hardware.sensors.interfaces import SensorInterface
from src.perception.detector import AluminiumCanDetector, DetectionResult


class CameraSensor(SensorInterface):
    """
    Concrete sensor that uses the C270 webcam + YOLOv8 to detect
    aluminium cans and expose their position to the Subsumption layers.

    Implements the SensorInterface contract:
        update()               — called every tick by main loop
        get_litter_position()  — returns (norm_x, norm_y) or None
    """

    def __init__(
        self,
        model_path: str = "ai/training/scripts/runs/detect/yolov8s_aluminum_can/weights/best.pt",
        camera_index: int = 0,
        conf_threshold: float = 0.5,
    ):
        self._detector = AluminiumCanDetector(
            model_path=model_path,
            camera_index=camera_index,
            conf_threshold=conf_threshold,
        )
        self._latest_result: DetectionResult = DetectionResult()
        self._battery: float = 1.0   # Placeholder; replace with real battery sensor
        self._obstacle: bool = False # Placeholder; replace with real proximity sensor

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Start the camera. Call once before the main loop."""
        self._detector.start()

    def stop(self):
        """Release camera. Call on system shutdown."""
        self._detector.stop()

    # ── SensorInterface Implementation ─────────────────────────────────

    def update(self):
        """
        Poll camera and run inference. Called every tick by main loop.
        Result is stored internally and queried by the Subsumption layers.
        """
        self._latest_result = self._detector.detect()

    def get_litter_position(self):
        """
        Returns normalized (x, y) of the best-detected aluminium can,
        where (0.5, 0.5) is the center of the frame.
        Returns None if no can is detected.

        Used by:
            layer1_scan.py    — to check if a target exists
            layer2_approach.py — to calculate motion vector towards the can
        """
        return self._latest_result.normalized_center()

    def get_aerial_trash_position(self):
        """Not used for floor litter. Returns None."""
        return None

    def get_battery_level(self) -> float:
        """Placeholder. Replace with actual battery sensor reading."""
        return self._battery

    def has_obstacle(self) -> bool:
        """Placeholder. Replace with actual proximity sensor reading."""
        return self._obstacle

    # ── Extra: Access raw result for debugging ──────────────────────────

    def get_latest_result(self) -> DetectionResult:
        """Returns the full DetectionResult from the last update() tick."""
        return self._latest_result

    def get_annotated_frame(self):
        """
        Returns the last camera frame with bounding boxes drawn.
        Use with cv2.imshow() for debugging.
        """
        return self._detector.get_annotated_frame(self._latest_result)
