# Utility helpers (logging, time calculations, notifications)
"""
Utility helpers used by escalation.

Includes:
- simple formatters
- safe extraction helpers
- small validation helpers
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def safe_get(d: Dict[str, Any], key: str, default=None):
    """Return d[key] if present and not None, otherwise default."""
    val = d.get(key, default)
    if val is None:
        return default
    return val


def format_target(target: Dict[str, Any]) -> str:
    """Produce a short human-readable description of a target."""
    ttype = target.get("type", "unknown")
    if ttype == "slack":
        return f"Slack(webhook={target.get('webhook')})"
    if ttype == "email":
        return f"Email(to={target.get('to_addrs')})"
    if ttype == "sms":
        return f"SMS(to={target.get('to_numbers')})"
    if ttype == "webhook":
        return f"Webhook(url={target.get('url')})"
    return f"{ttype}:{target}"
