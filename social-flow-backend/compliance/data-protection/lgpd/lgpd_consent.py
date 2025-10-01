"""
# Consent management engine for LGPD
"""
# compliance/data-protection/lgpd/lgpd_consent.py
"""
LGPD Consent Management
-----------------------
Implements Article 7 requirements for valid consent:
- Must be free, informed, and unequivocal
- Must specify purpose
- Can be revoked at any time
"""

import uuid
import datetime
from typing import Dict, List


class LGPDConsentManager:
    def __init__(self):
        self.consent_registry: Dict[str, List[Dict]] = {}

    def give_consent(self, user_id: str, purpose: str) -> str:
        """Register consent for a specific purpose."""
        token = str(uuid.uuid4())
        entry = {
            "id": token,
            "purpose": purpose,
            "granted_at": datetime.datetime.utcnow(),
            "revoked": False,
        }
        self.consent_registry.setdefault(user_id, []).append(entry)
        return token

    def revoke_consent(self, user_id: str, token: str) -> bool:
        """Revoke a previously granted consent."""
        for c in self.consent_registry.get(user_id, []):
            if c["id"] == token and not c["revoked"]:
                c["revoked"] = True
                c["revoked_at"] = datetime.datetime.utcnow()
                return True
        return False

    def has_valid_consent(self, user_id: str, purpose: str) -> bool:
        """Check if valid consent exists for a purpose."""
        return any(c for c in self.consent_registry.get(user_id, [])
                   if c["purpose"] == purpose and not c["revoked"])
