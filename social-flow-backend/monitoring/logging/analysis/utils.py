# Shared helper functions (parsing, serialization, etc.)
# monitoring/logging/analysis/utils.py
"""
Utility functions shared across log analysis modules.
Provides serialization, caching, datetime handling, and performance helpers.
"""

import json
import datetime
import hashlib
from functools import lru_cache


def serialize(obj):
    """Safely serialize Python objects to JSON string."""
    try:
        return json.dumps(obj, default=str, indent=2)
    except Exception as e:
        return f"SerializationError: {str(e)}"


def deserialize(json_str):
    """Safely deserialize JSON string to Python object."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def parse_timestamp(timestamp_str, formats):
    """Parse timestamp with multiple format attempts."""
    for fmt in formats:
        try:
            return datetime.datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    return None


@lru_cache(maxsize=1024)
def hash_event(event: str) -> str:
    """Return a stable hash for a log event string."""
    return hashlib.sha256(event.encode("utf-8")).hexdigest()
