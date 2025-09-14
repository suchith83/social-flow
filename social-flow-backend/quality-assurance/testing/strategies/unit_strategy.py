"""
Unit testing strategy module.

- Defines what unit tests must cover
- Metrics and gating rules (e.g., coverage thresholds)
- How to run locally vs CI
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from .config import STRATEGIES_CONFIG
from .exceptions import StrategyValidationError


@dataclass
class UnitStrategy:
    """
    Representation of unit testing policy and behavior.
    """
    required: bool = True
    coverage_threshold: float = field(default_factory=lambda: STRATEGIES_CONFIG.default_unit_coverage_threshold)
    required_tools: List[str] = field(default_factory=lambda: ["pytest", "coverage"])
    excluded_paths: List[str] = field(default_factory=list)
    notes: str = "Unit tests should be fast, deterministic, and mock external IO."

    def validate(self):
        if not (0 <= self.coverage_threshold <= 100):
            raise StrategyValidationError("coverage_threshold must be between 0 and 100")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "coverage_threshold": self.coverage_threshold,
            "required_tools": list(self.required_tools),
            "excluded_paths": list(self.excluded_paths),
            "notes": self.notes,
        }

    def summary(self) -> str:
        return f"UnitStrategy(required={self.required}, threshold={self.coverage_threshold}%)"
