# =========================
# File: testing/security/compliance/standards/gdpr_checker.py
# =========================
"""
GDPR-specific compliance checks
"""

from ..utils import scanner

def run_checks(policy_engine):
    evidence = {
        "user-data": scanner.scan_files_for_keywords(["delete", "erase", "remove"], "data/")
    }
    return policy_engine.validate("user-data", evidence)
