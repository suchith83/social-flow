"""
content_fingerprinter.py

Utilities for fingerprinting content to detect duplicates / near-duplicates.
Provides hashing utilities for text, images and video (frame hashing stub).
"""

import hashlib
import os
from typing import Tuple
from PIL import Image
import io
import numpy as np

# NOTE: PIL (Pillow) import used for basic image hashing. In production, use perceptual hashing (pHash) libs.


def sha256_bytes(data: bytes) -> str:
    """Deterministic fingerprint for arbitrary bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    """Fingerprint a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash_image_bytes(image_bytes: bytes, hash_size: int = 8) -> str:
    """
    Simple average hash (aHash) for image bytes:
    - Resize to hash_size x hash_size grayscale,
    - Compute mean and set bits accordingly.
    This is lightweight; replace with pHash/dHash for production.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(img).astype(np.float32)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    # Convert bits to hex string
    bit_string = "".join("1" if v else "0" for v in bits)
    return f"ahash-{int(bit_string, 2):x}"


def video_frame_hash_stub(video_path: str) -> Tuple[str, ...]:
    """
    Placeholder: sample frames, compute image hashes.
    In production use FFmpeg + frame sampling + perceptual hashing.
    Returns tuple of hashes for sampled frames.
    """
    # stub returns single sha256 of file for now
    return (sha256_file(video_path),)
