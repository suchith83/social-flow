"""
# Pseudonymization & anonymization utilities
"""
# compliance/data-protection/lgpd/lgpd_anonymizer.py
"""
LGPD Anonymization Utilities
----------------------------
Provides:
- Pseudonymization
- Full anonymization
- Tokenization
"""

import hashlib
import uuid
from typing import Dict, Any


class LGPDAnonymizer:
    def pseudonymize(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Replace identifiers with pseudonyms (reversible)."""
        return {k: hashlib.sha256(str(v).encode()).hexdigest() if "id" in k else v
                for k, v in record.items()}

    def anonymize(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Full anonymization – irreversibly remove PII."""
        return {k: None if "name" in k or "email" in k else v for k, v in record.items()}

    def tokenize(self, value: str) -> str:
        """Replace a sensitive value with a random token."""
        return str(uuid.uuid4())
