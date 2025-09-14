# Enrichment & normalization (masking, defaults)
"""
Enrichment and transformation of structured logs:
- add default service name, propagate trace/span ids, map legacy fields,
  normalize timestamp to UTC, mask PII if needed (simple example).
"""

from typing import Dict, Any
from datetime import datetime
from .config import CONFIG

MASK_FIELDS = {"password", "ssn", "credit_card", "secret"}


def enrich(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure required fields exist and normalize formats.
    Non-destructive: returns a new dict.
    """
    p = dict(payload)  # shallow copy
    if "service" not in p or not p.get("service"):
        p["service"] = CONFIG["DEFAULT_SERVICE_NAME"]

    # make sure timestamp is datetime or ISO string; allow StructuredLog to coerce
    ts = p.get("timestamp")
    if isinstance(ts, str):
        # try to parse ISO-ish string
        try:
            # preserve as ISO string; schema will convert
            # we leave unchanged here; StructuredLog will coerce
            pass
        except Exception:
            p["timestamp"] = datetime.utcnow()

    # normalize level
    lvl = p.get("level")
    if lvl:
        p["level"] = str(lvl).upper()

    # mask common PII
    for f in MASK_FIELDS:
        if f in p:
            p[f] = "***REDACTED***"

    # ensure attrs field exists
    p.setdefault("attrs", {})
    return p
