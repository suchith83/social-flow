# Helpers: batching, retries, formatting
"""
Utilities: batching, splitting lists, safe token chunking, basic metrics hooks.
"""

from typing import List, Iterable, Generator
import math


def chunked(iterable: Iterable, size: int) -> Generator[List, None, None]:
    """Yield successive `size`-sized chunks from iterable."""
    it = list(iterable)
    for i in range(0, len(it), size):
        yield it[i:i + size]


def validate_tokens(tokens):
    """Very light token validation; realistic projects need more checks."""
    valid = []
    for t in tokens:
        if not t or len(t) < 10:
            continue
        valid.append(t)
    return valid
