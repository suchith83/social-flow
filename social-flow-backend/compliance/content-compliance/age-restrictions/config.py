"""
config.py

Configurable thresholds and feature toggles for age restriction compliance.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ComplianceConfig:
    ENABLE_STRICT_MODE: bool = True
    ENABLE_AUDIT_LOGGING: bool = True
    DEFAULT_JURISDICTION: str = "GLOBAL"
    ESCALATION_THRESHOLD: int = 3  # number of violations before escalation
