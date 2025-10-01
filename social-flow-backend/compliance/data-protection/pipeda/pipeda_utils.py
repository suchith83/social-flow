"""
# Common helper functions (validation, formatting)
"""
# compliance/data-protection/pipeda/pipeda_utils.py
"""
PIPEDA Utility Helpers
----------------------
- Hashing
- ID generation
- Validation
"""

import hashlib
import re
import uuid


class PIPEDAUtils:
    @staticmethod
    def hash_value(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def validate_email(email: str) -> bool:
        return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())
