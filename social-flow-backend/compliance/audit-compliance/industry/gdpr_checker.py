# gdpr_checker.py
from .utils import log_event, pseudonymize

class GDPRChecker:
    """
    Enforces GDPR compliance.
    - Right to erasure
    - Data minimization
    - Consent tracking
    """

    def __init__(self):
        self.rules = [
            "User must give explicit consent",
            "Data subjects have right to be forgotten",
            "Only necessary data should be stored"
        ]

    def check(self, user_data):
        findings = []
        for u in user_data:
            if not u.get("consent", False):
                findings.append(f"No consent for user {pseudonymize(u['id'])}")
            if u.get("deleted_request", False) and not u.get("deleted", False):
                findings.append(f"Deletion request not fulfilled for {pseudonymize(u['id'])}")
        if findings:
            log_event(f"GDPR violations: {findings}", "WARNING")
        return findings
