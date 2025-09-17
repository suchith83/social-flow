"""
Validators for uploaded content
- Basic MIME/extension checks
- Size checks
- (Optional) heuristic checks like simple header sniff
"""

import os
import magic  # python-magic; note: in tests we will stub if unavailable
from .config import config
from .utils import file_hash, logger


class UploadValidationError(Exception):
    pass


def valid_extension(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


def valid_mime_type(mime: str) -> bool:
    if not mime:
        return False
    for p in config.ALLOWED_MIME_PREFIXES:
        if mime.startswith(p):
            return True
    return False


def validate_staged_file(path: str, expected_bytes: int = None, mime_override: str = None):
    # size check
    size = os.path.getsize(path)
    if expected_bytes and size != expected_bytes:
        raise UploadValidationError(f"Size mismatch: expected {expected_bytes}, got {size}")
    if size > config.MAX_UPLOAD_BYTES:
        raise UploadValidationError(f"File too large: {size} > {config.MAX_UPLOAD_BYTES}")
    # mime sniff
    mime = None
    try:
        mime = magic.from_file(path, mime=True)
    except Exception:
        # fallback: use provided override or skip
        mime = mime_override
    if not valid_mime_type(mime):
        raise UploadValidationError(f"Invalid mime type: {mime}")
    # extension check
    if not valid_extension(path):
        # allow by mime even if extension mismatches
        pass
    return {"size": size, "mime": mime, "sha256": file_hash(path)}
