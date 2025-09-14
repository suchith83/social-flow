# Shared helper utilities for parsing, masking, and retry logic
"""
Utility functions for security module.
Keep lightweight to ease unit testing.
"""

import ipaddress
import hashlib
import logging

logger = logging.getLogger("security_metrics.utils")

def parse_ip(ip_str: str):
    """Return an ipaddress.IPv4Address or IPv6Address or raise ValueError."""
    try:
        return ipaddress.ip_address(ip_str)
    except Exception as e:
        logger.debug("parse_ip: invalid ip [%s]: %s", ip_str, e)
        raise

def hash_event_id(*parts: str) -> str:
    """
    Deterministic hashed id for an event constructed from parts.
    Used for dedupe and correlate.
    """
    m = hashlib.sha256()
    for p in parts:
        if p is None:
            p = ""
        m.update(str(p).encode("utf-8"))
    return m.hexdigest()
