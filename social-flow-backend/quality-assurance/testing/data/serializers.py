"""
Serialization helpers for fixtures: JSON (compact and pretty), NDJSON, and optionally YAML.
"""

import json
from typing import Any, Iterable
from .utils import fingerprint


class Serializer:
    """Simple pluggable serializer."""

    def dumps(self, obj: Any, pretty: bool = False) -> str:
        """Serialize object to JSON string. Pretty prints when requested."""
        if pretty:
            return json.dumps(obj, indent=2, sort_keys=True, default=str)
        return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str)

    def dump_ndjson(self, iterable: Iterable[Any]) -> str:
        """Serialize an iterable to NDJSON (newline-delimited JSON)."""
        lines = []
        for obj in iterable:
            lines.append(json.dumps(obj, sort_keys=True, default=str))
        return "\n".join(lines)

    def fingerprint(self, obj: Any) -> str:
        return fingerprint(obj)
