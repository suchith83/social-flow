# key_builder.py
# Created by Create-Invalidation.ps1
"""
key_builder.py
---------------
Utilities to build and normalize cache keys consistently across the system.

Design goals:
- deterministic keys for the same logical object
- safe characters for caches and CDNs
- support namespaces, versioning, and key hashing for long keys
"""

from __future__ import annotations
import hashlib
import re
import json
from typing import Any, Iterable, Mapping, Optional

KEY_MAX_LEN = 250  # conservative limit for some caching systems


def _normalize_token(token: str) -> str:
    """Normalize a single token: lowercase, trim, replace whitespace and unsafe chars."""
    token = token.strip().lower()
    token = re.sub(r"\s+", "-", token)
    token = re.sub(r"[^a-z0-9\-_.:/]", "", token)
    return token


class CacheKeyBuilder:
    """
    Build cache keys in a consistent, safe way.

    Example usage:
        cb = CacheKeyBuilder(namespace="user-service", version="v2")
        key = cb.build("user", user_id=123, attrs={"active": True})
    """

    def __init__(
        self,
        namespace: Optional[str] = None,
        version: Optional[str] = None,
        max_len: int = KEY_MAX_LEN,
    ):
        self.namespace = _normalize_token(namespace) if namespace else None
        self.version = _normalize_token(version) if version else None
        self.max_len = max_len

    def _serialize_value(self, value: Any) -> str:
        """Deterministic JSON-ish serialization for dict-like parts."""
        if isinstance(value, Mapping):
            # sort keys for determinism
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        if isinstance(value, (list, tuple, set)):
            # convert to list and sort when possible
            try:
                return json.dumps(sorted(value), separators=(",", ":"))
            except Exception:
                return json.dumps(list(value), separators=(",", ":"))
        return str(value)

    def _hash_tail(self, text: str, length: int = 8) -> str:
        """Hash the tail of a text to keep key length manageable."""
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return h[:length]

    def build(self, base: str, *parts: Any, **kwparts: Any) -> str:
        """
        Build a key.

        - `base`: main resource token (e.g., "user", "timeline")
        - `parts`: positional tokens appended in order
        - `kwparts`: named tokens included as key=value pairs (deterministic sort)
        """
        if not base:
            raise ValueError("base token is required to build cache key")
        tokens = []
        if self.namespace:
            tokens.append(self.namespace)
        if self.version:
            tokens.append(self.version)

        tokens.append(_normalize_token(str(base)))

        # add positional parts
        for p in parts:
            tokens.append(_normalize_token(self._serialize_value(p)))

        # add kwparts in sorted order
        for k in sorted(kwparts.keys()):
            v = kwparts[k]
            token = f"{_normalize_token(str(k))}={_normalize_token(self._serialize_value(v))}"
            tokens.append(token)

        key = ":".join(tokens)

        if len(key) > self.max_len:
            # shorten deterministically by hashing the variable tail
            head = key[: self.max_len - 9]  # reserve 9 chars for sep+hash
            tail_hash = self._hash_tail(key)
            key = f"{head}:{tail_hash}"

        return key
