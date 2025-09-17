"""
Deduplication helpers.

Computes perceptual hash (pHash) and compares similarity.
Uses imagehash (Pillow-based) if available; otherwise falls back to average hash.
"""

from typing import Optional
import os
from PIL import Image
import imagehash
from .config import config
from .utils import logger


def compute_phash(image_path: str) -> Optional[str]:
    try:
        img = Image.open(image_path)
        ph = imagehash.phash(img)
        return str(ph)
    except Exception as e:
        logger.warning("pHash computation failed for %s: %s", image_path, e)
        return None


def phash_distance(a: str, b: str) -> int:
    """Return Hamming distance between two hex/hash strings using imagehash types."""
    try:
        ha = imagehash.hex_to_hash(a)
        hb = imagehash.hex_to_hash(b)
        return ha - hb
    except Exception:
        # fallback: treat as totally different
        return 100


def compute_phash_if_enabled(image_path: str) -> Optional[str]:
    if not config.ENABLE_PHASH:
        return None
    return compute_phash(image_path)
