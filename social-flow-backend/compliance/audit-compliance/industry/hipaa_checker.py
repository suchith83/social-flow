# hipaa_checker.py
from .utils import log_event, mask_sensitive

class HIPAAChecker:
    """
    Enforces HIPAA compliance rules.
    - Protects PHI (Protected Health Information).
    - Ensures access controls & minimal disclosure.
    """

    def __init__(self):
        self.rules = [
            "Access logs must be maintained",
            "PHI must be encrypted at rest and in transit",
            "Minimum necessary disclosure principle"
        ]

    def check(self, records):
        findings = []
        for rec in records:
            if not rec.get("encrypted", False):
                findings.append(f"PHI not encrypted: {mask_sensitive(rec.get('patient_id'))}")
            if not rec.get("access_log", False):
                findings.append("Missing access log entry")
        if findings:
            log_event(f"HIPAA violations found: {findings}", "ALERT")
        return findings
