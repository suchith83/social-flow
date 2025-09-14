"""
Integration testing strategy.

- Focuses on service interactions, DBs, messaging, third-party contracts
- Typically runs in CI or staging environments
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from .config import STRATEGIES_CONFIG
from .exceptions import StrategyValidationError


@dataclass
class IntegrationStrategy:
    required: bool = True
    coverage_threshold: float = field(default_factory=lambda: STRATEGIES_CONFIG.default_integration_coverage_threshold)
    environments: List[str] = field(default_factory=lambda: ["ci", "staging"])
    dependencies: List[str] = field(default_factory=lambda: ["postgres", "redis"])
    flakiness_tolerance: float = 0.02  # acceptable flaky rate measured over time
    notes: str = "Integration tests should exercise components and ports but avoid full UI rendering."

    def validate(self):
        if not (0 <= self.coverage_threshold <= 100):
            raise StrategyValidationError("coverage_threshold must be between 0 and 100")
        if not (0 <= self.flakiness_tolerance <= 1):
            raise StrategyValidationError("flakiness_tolerance must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "coverage_threshold": self.coverage_threshold,
            "environments": list(self.environments),
            "dependencies": list(self.dependencies),
            "flakiness_tolerance": self.flakiness_tolerance,
            "notes": self.notes,
        }

    def summary(self) -> str:
        return f"IntegrationStrategy(required={self.required}, threshold={self.coverage_threshold}%)"
