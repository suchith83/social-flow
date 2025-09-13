# Utility functions (logging, batching, serialization)
"""
Utility helpers used across the module.
"""

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict
from .config import get_config

config = get_config()


def parse_iso_datetime(s: str) -> datetime:
    # Very permissive ISO parser; production use dateutil.parser.isoparse
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def anonymize_payload(payload: dict) -> dict:
    """
    Remove or mask PII fields from event payloads when ANONYMIZE_PII is true.
    This function should be customized to the app's schema.
    """
    if not config.ANONYMIZE_PII:
        return payload
    cleaned = {}
    for k, v in payload.items():
        if k.lower() in ("email", "phone", "ssn", "name"):
            cleaned[k] = "<redacted>"
        else:
            cleaned[k] = v
    return cleaned


def size_of_json(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def metric_bucket_for(dt: datetime, interval_seconds: int):
    """
    Compute bucket start/end for a timestamp and interval.
    Returns (start_datetime, end_datetime).
    """
    ts = int(dt.replace(tzinfo=timezone.utc).timestamp())
    start_ts = ts - (ts % interval_seconds)
    end_ts = start_ts + interval_seconds
    return datetime.fromtimestamp(start_ts, tz=timezone.utc), datetime.fromtimestamp(end_ts, tz=timezone.utc)
