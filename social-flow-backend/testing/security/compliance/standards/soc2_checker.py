# =========================
# File: testing/security/compliance/standards/soc2_checker.py
# =========================
"""
SOC2-specific compliance checks
"""

from ..utils import scanner

def run_checks(policy_engine):
    evidence = {
        "auth-service": scanner.scan_logs_for_pattern("logs/auth.log", "failed login"),
        "logging": {"contains_errors": False}
    }
    return policy_engine.validate("auth-service", evidence) + \
           policy_engine.validate("logging", evidence)
