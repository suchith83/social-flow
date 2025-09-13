# Predefined rule templates for reuse
"""
Lightweight templating utilities.

We use Python's str.format-style templates with safe replacement,
plus optional jinja2 if available. Defaults to a small, safe renderer.
"""

from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

try:
    from jinja2 import Template, Environment, select_autoescape
    JINJA_AVAILABLE = True
except Exception:
    JINJA_AVAILABLE = False

def _safe_merge_context(rule, event: Dict[str, Any], matched: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rule": {"id": rule.id, "name": rule.name, "description": rule.description},
        "event": event,
        "matched": matched,
    }

def render_template(template: Optional[str], rule, event: Dict[str, Any], matched: Dict[str, Any]) -> str:
    """
    Render a template string using jinja2 if available, otherwise with basic python formatting.
    Provide `rule`, `event`, and `matched` in the template context.
    """
    ctx = _safe_merge_context(rule, event, matched)
    if not template:
        # default representation
        try:
            return f"[{rule.name}] {json.dumps(event, default=str)}"
        except Exception:
            return f"[{rule.name}] {str(event)}"

    if JINJA_AVAILABLE:
        env = Environment(autoescape=select_autoescape(enabled_extensions=()))
        t = env.from_string(template)
        return t.render(**ctx)
    else:
        # fallback: allow simple {rule[name]} and {event[foo]} style usage via format_map
        # Build a flattened context of string keys
        flat = {
            "rule.name": rule.name,
            "rule.id": rule.id,
            "rule.description": rule.description,
            "event": json.dumps(event, default=str),
            "matched": json.dumps(matched, default=str),
        }
        # naive replacement: replace patterns like {{rule.name}} or {rule.name}
        out = template
        for k, v in flat.items():
            out = out.replace("{{" + k + "}}", str(v)).replace("{" + k + "}", str(v))
        return out
