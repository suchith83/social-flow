"""
Utility helpers for analytics-data
"""

import datetime
import hashlib
import json
import logging
from typing import Any, Dict

from .config import config

logger = logging.getLogger("analytics-data")
logger.setLevel(config.LOG_LEVEL)


def utc_now() -> datetime.datetime:
    """Return current UTC datetime"""
    return datetime.datetime.utcnow()


def json_hash(data: Dict[str, Any]) -> str:
    """Return stable hash for JSON object (for deduplication)"""
    encoded = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe_json(data: Any) -> str:
    """Serialize safely to JSON string"""
    try:
        return json.dumps(data, default=str)
    except Exception as e:
        logger.error(f"Failed JSON serialization: {e}")
        return "{}"
