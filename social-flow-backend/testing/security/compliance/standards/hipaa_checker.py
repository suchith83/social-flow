# =========================
# File: testing/security/compliance/standards/hipaa_checker.py
# =========================
"""
HIPAA-specific compliance checks
"""

from ..utils import scanner

def run_checks(policy_engine):
    evidence = {
        "database": scanner.check_encryption("db/config.yaml"),
        "api": scanner.scan_endpoints_for_tls("https://api.local/health")
    }
    return policy_engine.validate("database", evidence) + \
           policy_engine.validate("api", evidence)
