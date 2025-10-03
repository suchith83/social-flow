"""
Copyright models package.

This module exports copyright-related models.
"""

from app.copyright.models.copyright_fingerprint import (
    Copyright,
    CopyrightFingerprint,
    CopyrightMatch,
)

# Aliases for backward compatibility
CopyrightClaim = Copyright
ContentFingerprint = CopyrightFingerprint

__all__ = [
    "Copyright",
    "CopyrightClaim",
    "CopyrightFingerprint",
    "ContentFingerprint",
    "CopyrightMatch",
]
