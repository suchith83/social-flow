# Config management for load balancer
# performance/scaling/load_balancing/config.py

import yaml
import logging
from typing import Dict, Any


class Config:
    """
    Load balancing configuration management.
    """

    DEFAULTS = {
        "balancer": {
            "algorithm": "round_robin",
            "healthcheck_interval": 10,
            "nodes": [
                {"host": "127.0.0.1", "port": 8001, "weight": 1},
                {"host": "127.0.0.1", "port": 8002, "weight": 2},
            ],
        },
        "logging": {"level": "INFO"},
    }

    @classmethod
    def load(cls, path: str) -> Dict[str, Any]:
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
        result = default.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in result:
                result[k] = Config._deep_merge(result[k], v)
            else:
                result[k] = v
        return result
