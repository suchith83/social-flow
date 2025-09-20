"""Small utilities used by services when shared common packages are not installed.

Provides:
- get_logger(name): lightweight logging setup with optional JSON formatting via env var SF_LOG_JSON
- Config: minimal env-backed config accessor
"""
from typing import Optional
import logging
import os
import json

LOG_JSON = os.environ.get("SF_LOG_JSON", "false").lower() in ("1", "true", "yes")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload)
        except Exception:
            return str(payload)


def get_logger(name: str, level: Optional[int] = logging.INFO) -> logging.Logger:
    """Return a configured logger. Safe to call multiple times."""
    l = logging.getLogger(name)
    if not l.handlers:
        h = logging.StreamHandler()
        if LOG_JSON:
            h.setFormatter(JsonFormatter())
        else:
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            h.setFormatter(fmt)
        l.addHandler(h)
    l.setLevel(level)
    return l


class Config:
    """Minimal configuration helper reading from environment variables."""

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = os.environ.get(key)
        if val is None:
            return default
        return val.lower() in ("1", "true", "yes")

    def get_int(self, key: str, default: int = 0) -> int:
        val = os.environ.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except Exception:
            return default
