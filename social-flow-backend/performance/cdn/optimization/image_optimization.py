# Optimizes images for faster delivery
# performance/cdn/optimization/image_optimization.py
"""
Image optimization strategies.
"""

from io import BytesIO
from PIL import Image
from typing import Optional
from .utils import logger, content_hash

class ImageOptimizer:
    def __init__(self, default_quality: int = 80):
        self.default_quality = default_quality

    def resize(self, img_bytes: bytes, width: int, height: int) -> bytes:
        """Resize image and return new bytes."""
        with Image.open(BytesIO(img_bytes)) as img:
            img = img.resize((width, height))
            out = BytesIO()
            img.save(out, format=img.format, quality=self.default_quality)
            data = out.getvalue()
            logger.debug(f"Resized image {len(img_bytes)}B -> {len(data)}B hash={content_hash(data)}")
            return data

    def convert_format(self, img_bytes: bytes, fmt: str = "WEBP") -> bytes:
        """Convert to another format (e.g., WEBP, AVIF)."""
        with Image.open(BytesIO(img_bytes)) as img:
            out = BytesIO()
            img.save(out, format=fmt, quality=self.default_quality)
            data = out.getvalue()
            logger.debug(f"Converted image to {fmt} size={len(data)}B hash={content_hash(data)}")
            return data
