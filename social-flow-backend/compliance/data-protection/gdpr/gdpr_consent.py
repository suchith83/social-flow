"""
# Consent management utilities
"""
"""
GDPR Consent Management Service
-------------------------------
Handles user consent collection and withdrawal.
"""

from datetime import datetime, timedelta
from .gdpr_exceptions import GDPRConsentError

class GDPRConsentService:
    """Manages user consent records."""

    def __init__(self):
        self.consent_registry = {}  # user_id -> {"consent": bool, "timestamp": datetime}

    def give_consent(self, user_id: str):
        try:
            self.consent_registry[user_id] = {"consent": True, "timestamp": datetime.utcnow()}
            return {"status": "consent_given"}
        except Exception as e:
            raise GDPRConsentError(f"Failed to record consent: {e}")

    def withdraw_consent(self, user_id: str):
        try:
            self.consent_registry[user_id] = {"consent": False, "timestamp": datetime.utcnow()}
            return {"status": "consent_withdrawn"}
        except Exception as e:
            raise GDPRConsentError(f"Failed to withdraw consent: {e}")

    def has_valid_consent(self, user_id: str, retention_days: int = 1825) -> bool:
        record = self.consent_registry.get(user_id)
        if not record:
            return False
        expiry = record["timestamp"] + timedelta(days=retention_days)
        return record["consent"] and datetime.utcnow() <= expiry
