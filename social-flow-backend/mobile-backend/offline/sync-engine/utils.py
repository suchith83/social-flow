# Utility helpers (diffing, batching, retries)
"""
Utility helpers for sync engine: key normalization, validation, and lightweight transforms.
"""

import re
from typing import Optional

_key_pattern = re.compile(r"^[a-zA-Z0-9_\-:]+$")


def normalize_key(key: str) -> str:
    k = key.strip()
    if not _key_pattern.match(k):
        # fallback: replace spaces with dash
        k = re.sub(r"\s+", "-", k)
    return k.lower()


def validate_key(key: str) -> bool:
    return bool(_key_pattern.match(key))
