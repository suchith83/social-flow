# Configuration management (thresholds, scaling rules)
# performance/scaling/auto_scaling/config.py

import yaml
import logging
from typing import Dict, Any


class Config:
    """
    Configuration management for auto-scaling.

    Supports YAML configuration with defaults and validation.
    """

    DEFAULTS = {
        "scaling": {
            "min_instances": 1,
            "max_instances": 20,
            "cooldown_seconds": 120,
            "metrics": ["cpu", "memory"],
            "policy": "threshold"
        },
        "logging": {
            "level": "INFO"
        }
    }

    @classmethod
    def load(cls, path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with defaults applied.
        """
        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {path}")

        merged = cls._deep_merge(cls.DEFAULTS, raw)
        logging.basicConfig(level=merged["logging"]["level"])
        return merged

    @staticmethod
    def _deep_merge(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries (default <- override).
        """
        result = default.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in result:
                result[k] = Config._deep_merge(result[k], v)
            else:
                result[k] = v
        return result
