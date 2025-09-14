# Configuration for regions & policies
# performance/scaling/regional/config.py

import yaml
import logging
from typing import Dict, Any


class Config:
    """
    Configuration management for regional scaling.
    """

    DEFAULTS = {
        "regions": {
            "us-east": {"endpoint": "us-east.example.com", "weight": 3},
            "eu-west": {"endpoint": "eu-west.example.com", "weight": 2},
            "ap-south": {"endpoint": "ap-south.example.com", "weight": 1},
        },
        "policy": "latency",  # options: latency, geo, weighted, failover
        "healthcheck_interval": 15,
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
