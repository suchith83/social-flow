"""
Fixture loaders for JSON and NDJSON files. Provides safe loading with validation hooks.
"""

import json
from typing import Any, Iterable, List
import os
from .exceptions import FixtureLoadError
from .utils import fingerprint


class FixtureLoader:
    """Load fixtures from disk."""

    def load(self, path: str) -> List[Any]:
        """Load JSON or NDJSON from path and return list of objects."""
        if not os.path.exists(path):
            raise FixtureLoadError(f"Fixture path does not exist: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext in (".json", ""):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Always return list semantics for fixtures (single object -> [obj])
            if isinstance(data, list):
                return data
            return [data]
        else:
            # Support ndjson (.ndjson or .jsonl)
            return self._load_ndjson(path)

    def _load_ndjson(self, path: str) -> List[Any]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise FixtureLoadError(f"Invalid JSON in {path}: {e}")
        return items
