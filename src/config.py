from __future__ import annotations

from pathlib import Path

import yaml


class AppConfig:
    """
    Handles loading and accessing configuration from a .yaml file.
    Supports dot-notation for nested key access.
    """

    def __init__(self, config_path: str | Path = "config.yaml"):
        if not isinstance(config_path, Path):
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def _get_nested(self, keys: str) -> any:
        """Helper to access nested dictionary keys using dot-notation."""
        value = self.config
        for key in keys.split("."):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def get(self, key: str, fallback: str | None = None) -> str | None:
        """Gets a string value from the config. Use dot-notation for nested keys."""
        value = self._get_nested(key)
        return str(value) if value is not None else fallback

    def getint(self, key: str, fallback: int | None = None) -> int | None:
        """Gets an integer value from the config. Use dot-notation for nested keys."""
        value = self._get_nested(key)
        return int(value) if value is not None else fallback

    def getfloat(self, key: str, fallback: float | None = None) -> float | None:
        """Gets a float value from the config. Use dot-notation for nested keys."""
        value = self._get_nested(key)
        return float(value) if value is not None else fallback

    def getboolean(self, key: str, fallback: bool | None = None) -> bool | None:
        """Gets a boolean value from the config. Use dot-notation for nested keys."""
        value = self._get_nested(key)
        return bool(value) if value is not None else fallback


# Global config instance to be used across the application
try:
    config = AppConfig()
except FileNotFoundError as e:
    print(f"Warning: {e}. Using default values.")
    config = None

# --- Hand Raise Detection ---
HAND_RAISE_SECONDS_THRESHOLD = config.getfloat("hand_raise.seconds_threshold", 1.0)
WRIST_SHOULDER_VERTICAL_THRESHOLD = config.getint("hand_raise.wrist_shoulder_vertical_threshold", -20)
