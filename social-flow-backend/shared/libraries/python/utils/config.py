# config.py
import os
from typing import Optional

# Try to load .env if available (helpful in local dev)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv not installed — ignore
    pass


class Config:
    """Minimal configuration helper for services.

    Usage: cfg = Config(); cfg.get("KEY", default)
    """

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
