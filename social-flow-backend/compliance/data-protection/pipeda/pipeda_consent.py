"""
# Consent collection and withdrawal utilities
"""
# compliance/data-protection/pipeda/pipeda_consent.py
"""
PIPEDA Consent Management
-------------------------
- Organizations must obtain meaningful consent
- Must explain purposes
- Allow withdrawal
"""

import uuid
import datetime
from typing import Dict, List


class PIPEDAConsentManager:
    def __init__(self):
        self.registry: Dict[str, List[Dict]] = {}

    def obtain_consent(self, user_id: str, purpose: str, description: str) -> str:
        """Obtain consent from individual."""
        token = str(uuid.uuid4())
        entry = {
            "id": token,
            "purpose": purpose,
            "description": description,
            "obtained_at": datetime.datetime.utcnow(),
            "withdrawn": False,
        }
        self.registry.setdefault(user_id, []).append(entry)
        return token

    def withdraw_consent(self, user_id: str, token: str) -> bool:
        """Withdraw consent."""
        for c in self.registry.get(user_id, []):
            if c["id"] == token and not c["withdrawn"]:
                c["withdrawn"] = True
                c["withdrawn_at"] = datetime.datetime.utcnow()
                return True
        return False

    def has_consent(self, user_id: str, purpose: str) -> bool:
        """Check valid consent."""
        return any(c for c in self.registry.get(user_id, [])
                   if c["purpose"] == purpose and not c["withdrawn"])
