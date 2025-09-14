"""
End-to-end (E2E) testing strategy.

- Runs against deployed staging environments or local docker-compose environments.
- Include test timing, retries, network/latency simulation guidance.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from .config import STRATEGIES_CONFIG
from .exceptions import StrategyValidationError


@dataclass
class E2EStrategy:
    required: bool = False  # E2E often heavier and not always run per PR
    timeout_seconds: int = field(default_factory=lambda: STRATEGIES_CONFIG.default_e2e_timeout_seconds)
    parallel_workers: int = 3
    flaky_retry_policy: Dict[str, int] = field(default_factory=lambda: {"retries": 1, "backoff_seconds": 5})
    browsers: List[str] = field(default_factory=lambda: ["chromium", "firefox"])
    notes: str = "E2E tests are integration tests from a user's perspective. Keep them high-value."

    def validate(self):
        if self.timeout_seconds <= 0:
            raise StrategyValidationError("timeout_seconds must be positive")
        if self.parallel_workers < 1:
            raise StrategyValidationError("parallel_workers must be >= 1")
        if self.flaky_retry_policy.get("retries", 0) < 0:
            raise StrategyValidationError("retries must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "timeout_seconds": self.timeout_seconds,
            "parallel_workers": self.parallel_workers,
            "flaky_retry_policy": dict(self.flaky_retry_policy),
            "browsers": list(self.browsers),
            "notes": self.notes,
        }

    def summary(self) -> str:
        return f"E2EStrategy(required={self.required}, timeout={self.timeout_seconds}s)"
