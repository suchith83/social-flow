"""
# Policy definitions and obligations under CCPA
"""
"""
CCPA Policy Definitions
-----------------------
Defines policies and obligations for organizations to comply 
with the California Consumer Privacy Act.
"""

from datetime import timedelta, datetime

class CCPAPolicy:
    """Encapsulates CCPA policy definitions and obligations."""

    DATA_ACCESS_WINDOW = timedelta(days=45)  # Must respond within 45 days
    DATA_DELETION_WINDOW = timedelta(days=45)
    OPT_OUT_VALIDITY = timedelta(days=365)

    @staticmethod
    def user_rights():
        """Return a dictionary of consumer rights under CCPA."""
        return {
            "Right to Know": "Consumers have the right to request disclosure of the personal information collected.",
            "Right to Delete": "Consumers can request deletion of their personal data.",
            "Right to Opt-Out": "Consumers can opt-out of the sale of their personal information.",
            "Right to Non-Discrimination": "Consumers cannot be discriminated against for exercising their rights."
        }

    @staticmethod
    def requires_verification(request_type: str) -> bool:
        """
        Some requests require user identity verification.
        """
        return request_type in {"access", "deletion"}

    @staticmethod
    def is_within_deadline(request_date: datetime, request_type: str) -> bool:
        """
        Check if a request is still within the legally mandated response window.
        """
        now = datetime.utcnow()
        deadline = {
            "access": CCPAPolicy.DATA_ACCESS_WINDOW,
            "deletion": CCPAPolicy.DATA_DELETION_WINDOW,
            "optout": CCPAPolicy.OPT_OUT_VALIDITY,
        }.get(request_type, timedelta(days=0))
        return now <= request_date + deadline
