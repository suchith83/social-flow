"""
Utility helpers for the strategies package.
"""

import logging
import json
from typing import Any, Dict, Iterable

logger = logging.getLogger("qa-testing-strategies")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def safe_dumps(obj: Any, pretty: bool = False) -> str:
    """Serialize an object to JSON safely (fallbacks to str)."""
    try:
        if pretty:
            return json.dumps(obj, indent=2, sort_keys=True, default=str)
        return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str)
    except Exception:
        logger.exception("Failed to JSON serialize object; using repr.")
        return repr(obj)


def ensure_list(val):
    """Normalize a value into a list (if None -> empty list, if str -> [str])."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return list(val)
    return [val]
