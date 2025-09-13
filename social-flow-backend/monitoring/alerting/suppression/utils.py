# Utility helpers (logging, time calculations, validation)
"""
Utility helpers used by suppression.

- extract_path: safe dotted-path extraction from nested dicts
- now_utc: timezone-aware UTC timestamp
- seconds_to_iso: convert seconds to ISO-like human readable duration string
"""

from typing import Any, Dict, Optional
from datetime import datetime, timezone, timedelta


def extract_path(obj: Dict[str, Any], dotted_path: str) -> Optional[Any]:
    """
    Safely extract a nested value from a dict using dotted path.
    Returns None if path does not exist.
    """
    if not dotted_path:
        return None
    cur = obj
    for p in dotted_path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def now_utc() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def seconds_to_iso(seconds: Optional[int]) -> Optional[str]:
    """Return ISO 8601-like duration or None."""
    if seconds is None:
        return None
    td = timedelta(seconds=int(seconds))
    # naive formatting like "PT5M" (only supports seconds->hours/minutes)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}H")
    if minutes:
        parts.append(f"{minutes}M")
    if secs:
        parts.append(f"{secs}S")
    if not parts:
        parts.append("0S")
    return "PT" + "".join(parts)
