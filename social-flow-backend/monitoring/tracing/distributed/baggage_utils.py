# baggage_utils.py
# Created by Create-DistributedFiles.ps1
"""
Baggage helpers for lightweight, low-cardinality key:value propagation across services.

Provides:
 - set_baggage/get_baggage convenience wrappers
 - propagate_baggage_headers to convert baggage into headers for systems that require it
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger("tracing.distributed.baggage_utils")

try:
    from opentelemetry.baggage import set_baggage as otel_set_baggage, get_all as otel_get_all
    from opentelemetry.propagate import inject
    BAGGAGE_AVAILABLE = True
except Exception:
    BAGGAGE_AVAILABLE = False
    logger.debug("OpenTelemetry baggage not available.")


def set_baggage(key: str, value: str):
    """Set baggage key in current context."""
    if not BAGGAGE_AVAILABLE:
        return
    try:
        otel_set_baggage(key, value)
    except Exception:
        logger.exception("set_baggage failed for %s", key)


def get_baggage(key: str) -> Optional[str]:
    """Return baggage value or None."""
    if not BAGGAGE_AVAILABLE:
        return None
    try:
        all_b = otel_get_all()
        return all_b.get(key)
    except Exception:
        logger.exception("get_baggage failed for %s", key)
        return None


def propagate_baggage_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject baggage into headers using OTel inject (best-effort).
    Returns headers dict mutated.
    """
    if not BAGGAGE_AVAILABLE:
        return headers
    try:
        inject(headers)
    except Exception:
        logger.exception("propagate_baggage_headers failed")
    return headers
