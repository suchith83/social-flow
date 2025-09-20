"""Tiny schema validator wrapper. Uses jsonschema if installed, otherwise no-op."""

from typing import Any, Dict, Optional

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None  # type: ignore


def validate_json(schema: Dict[str, Any], payload: Dict[str, Any]) -> Optional[str]:
    """
    Validate payload against schema. Returns None if valid, otherwise returns error message.
    This is a convenience helper — use your proper schema registry in production.
    """
    if jsonschema is None:
        return None
    try:
        jsonschema.validate(instance=payload, schema=schema)
        return None
    except Exception as exc:
        return str(exc)
