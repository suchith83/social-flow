# Utility functions (serialization, batching, TTL helpers)
"""
Utility helpers: checksums, key normalization, http streaming helper for fetching remote content.
"""

import hashlib
import requests
from typing import Tuple, Optional, Iterable, Generator

def sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def normalize_key(key: str) -> str:
    # Basic normalization: strip whitespace, lower-case. In real systems consider namespace prefixes.
    return key.strip()


def stream_fetch(url: str, chunk_size: int = 64 * 1024) -> Generator[bytes, None, None]:
    """
    Stream remote content in chunks using requests.
    Yields bytes chunks. Caller is responsible for assembling or writing to disk.
    """
    with requests.get(url, stream=True, timeout=20) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk
