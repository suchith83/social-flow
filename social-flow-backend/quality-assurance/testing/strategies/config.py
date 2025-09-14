"""
Configuration and defaults for testing strategies.
"""

import os
from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class StrategiesConfig:
    """
    Configurable defaults that can be overridden through environment variables.
    Keep values conservative and sensible for CI gating.
    """
    default_unit_coverage_threshold: float = float(os.getenv("UNIT_COVERAGE_THRESHOLD", "90.0"))
    default_integration_coverage_threshold: float = float(os.getenv("INTEGRATION_COVERAGE_THRESHOLD", "80.0"))
    default_e2e_timeout_seconds: int = int(os.getenv("E2E_TIMEOUT", "600"))
    allowed_test_levels: List[str] = ("unit", "integration", "e2e", "contract")
    ci_required_checks: List[str] = ("unit-coverage","integration-coverage","smoke")
    max_risk_level: str = os.getenv("MAX_RISK_LEVEL", "medium")  # one of low/medium/high

    # Mapping which test types should run where (local/dev/ci/prod)
    default_matrix: Dict[str, List[str]] = None

    def __post_init__(self):
        # dataclass frozen workaround to set default matrix if not provided
        if self.default_matrix is None:
            object.__setattr__(self, "default_matrix", {
                "local": ["unit"],
                "ci": ["unit", "integration", "contract"],
                "staging": ["integration", "e2e", "contract"],
                "production": ["smoke"]
            })


STRATEGIES_CONFIG = StrategiesConfig()
