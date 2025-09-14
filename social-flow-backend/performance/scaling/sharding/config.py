# Sharding configuration
# performance/scaling/sharding/config.py

import yaml
import logging
from typing import Dict, Any


class Config:
    """
    Configuration management for sharding system.
    """

    DEFAULTS = {
        "shards": {
            "shard_1": {"node": "db1.example.com", "range": [0, 5000]},
            "shard_2": {"node": "db2.example.com", "range": [5001, 10000]},
        },
        "algorithm": "hash",  # options: hash, range, consistent_hash
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
