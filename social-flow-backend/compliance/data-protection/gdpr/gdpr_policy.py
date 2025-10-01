"""
# Policy definitions and obligations for GDPR
"""
"""
GDPR Policy Definitions
-----------------------
Defines organizational obligations for GDPR compliance.
"""

from datetime import timedelta, datetime

class GDPRPolicy:
    """Encapsulates GDPR policy definitions."""

    DATA_ACCESS_WINDOW = timedelta(days=30)   # GDPR: 1 month response time
    DATA_DELETION_WINDOW = timedelta(days=30)
    CONSENT_RETENTION = timedelta(days=365*5) # Keep consent logs for 5 years

    @staticmethod
    def user_rights():
        return {
            "Right of Access": "Individuals can request access to their personal data.",
            "Right to Rectification": "Individuals can request corrections to their data.",
            "Right to Erasure": "Individuals can request deletion (right to be forgotten).",
            "Right to Restriction": "Processing can be limited in specific cases.",
            "Right to Data Portability": "Individuals can request data in a portable format.",
            "Right to Object": "Individuals can object to processing (e.g., direct marketing).",
            "Rights related to automated decision-making": "Individuals can refuse profiling/AI-only decisions."
        }

    @staticmethod
    def is_within_deadline(request_date: datetime, request_type: str) -> bool:
        now = datetime.utcnow()
        deadline = {
            "access": GDPRPolicy.DATA_ACCESS_WINDOW,
            "deletion": GDPRPolicy.DATA_DELETION_WINDOW,
            "portability": GDPRPolicy.DATA_ACCESS_WINDOW,
        }.get(request_type, timedelta(days=0))
        return now <= request_date + deadline
