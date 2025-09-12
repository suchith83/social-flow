"""
# Common helpers (validators, crypto, localization)
"""
# compliance/data-protection/lgpd/lgpd_utils.py
"""
LGPD Utility Helpers
--------------------
Provides:
- Cryptographic hashing
- Localization utilities
- Validation
"""

import hashlib
import re
import uuid


class LGPDUtils:
    @staticmethod
    def hash_value(value: str) -> str:
        """Return SHA-256 hash of value."""
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

    @staticmethod
    def generate_request_id() -> str:
        """Generate unique request identifier."""
        return str(uuid.uuid4())
