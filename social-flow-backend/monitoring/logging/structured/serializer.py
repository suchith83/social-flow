# Serialization (to/from JSON + NDJSON)
"""
Serialization helpers: object -> JSON string / NDJSON line and back.
"""

from typing import Dict, Any
from .schema import StructuredLog
from .utils import safe_serialize
from .config import CONFIG


def to_json(obj: Dict[str, Any], ensure_ascii: bool = None, indent=None) -> str:
    ea = CONFIG["SERIALIZATION"]["ensure_ascii"] if ensure_ascii is None else ensure_ascii
    indent = CONFIG["SERIALIZATION"]["indent"] if indent is None else indent
    # convert StructuredLog model to dict if needed
    if isinstance(obj, StructuredLog):
        obj_dict = obj.dict()
    else:
        obj_dict = obj
    return safe_serialize(obj_dict, ensure_ascii=ea, indent=indent)


def to_ndjson_line(obj: Dict[str, Any]) -> str:
    """Produce a single-line JSON string (ndjson)."""
    return to_json(obj, indent=None)


def from_json_str(s: str) -> Dict[str, Any]:
    import json
    return json.loads(s)
