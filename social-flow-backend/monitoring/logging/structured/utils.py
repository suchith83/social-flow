# Helpers (safe serialization, NDJSON batching, etc.)
"""
Utilities for structured logging:
- safe_serialize: guard against unserializable objects
- ndjson batching helpers
- helper to enforce small memory footprint when writing logs
"""

import json
from typing import Any, Iterable, List
from .config import CONFIG


def safe_serialize(obj: Any, ensure_ascii: bool = False, indent=None) -> str:
    """
    Serialize object to JSON safely; fallback to str() on failure for problematic fields.
    """
    try:
        return json.dumps(obj, ensure_ascii=ensure_ascii, default=_fallback, indent=indent)
    except Exception:
        # Extremely defensive fallback: stringify
        return json.dumps(str(obj))


def _fallback(o):
    # permitted fallback for common problematic types
    try:
        return str(o)
    except Exception:
        return "<unserializable>"

def ndjson_batches(iterable: Iterable[str], batch_size: int = None):
    """
    Yield lists of ndjson lines in batches.
    """
    if batch_size is None:
        batch_size = CONFIG["SERIALIZATION"]["ndjson_batch_size"]
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
