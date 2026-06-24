"""
Configuration management for AutoBin.
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv


class Config:
    """Manages configuration for the AutoBin system."""

    def __init__(self, config_file: str = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to YAML configuration file
        """
        # Load environment variables
        load_dotenv()

        # Default configuration
        self.config: Dict[str, Any] = self._get_defaults()

        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

        # Override with environment variables
        self._load_from_env()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "camera": {
                "index": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "detection": {
                "model": "yolov8n.pt",
                "confidence": 0.5,
                "iou_threshold": 0.45,
                "device": "cpu"
            },
            "movement": {
                "max_speed": 1.0,
                "obstacle_distance": 0.3,
                "safety_margin": 0.2
            },
            "hardware": {
                "motor_left_pwm": 12,
                "motor_right_pwm": 13,
                "ultrasonic_trig": 23,
                "ultrasonic_echo": 24
            },
            "logging": {
                "level": "INFO",
                "file": "logs/autobin.log"
            }
        }

    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            file_config = yaml.safe_load(f)
            self._merge_config(file_config)

    def _merge_config(self, new_config: Dict[str, Any]):
        """Recursively merge new configuration into existing."""
        def merge_dict(base: dict, update: dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value

        merge_dict(self.config, new_config)

    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mapping = {
            "CAMERA_INDEX": ("camera", "index", int),
            "CAMERA_WIDTH": ("camera", "width", int),
            "CAMERA_HEIGHT": ("camera", "height", int),
            "DETECTION_CONFIDENCE": ("detection", "confidence", float),
            "DETECTION_MODEL": ("detection", "model", str),
            "MAX_SPEED": ("movement", "max_speed", float),
            "LOG_LEVEL": ("logging", "level", str),
        }

        for env_var, (section, key, type_func) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                self.config[section][key] = type_func(value)

    def get(self, *keys, default=None) -> Any:
        """
        Get configuration value by nested keys.

        Args:
            *keys: Nested keys to access value
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            config.get("camera", "width")  # Returns camera width
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys, value):
        """
        Set configuration value by nested keys.

        Args:
            *keys: Nested keys to access location
            value: Value to set

        Example:
            config.set("camera", "width", value=1280)
        """
        target = self.config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def save(self, config_file: str):
        """Save configuration to YAML file."""
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
