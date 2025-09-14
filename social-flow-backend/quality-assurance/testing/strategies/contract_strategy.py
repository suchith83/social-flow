"""
Contract testing strategy (consumer-driven contract, schema-based checks).

- Use Pact, contract tests, and schema validations to prevent contract regressions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from .exceptions import StrategyValidationError


@dataclass
class ContractStrategy:
    required: bool = True
    providers: List[str] = field(default_factory=list)
    consumers: List[str] = field(default_factory=list)
    tooling: List[str] = field(default_factory=lambda: ["pact", "schemathesis"])
    publish_location: str = "contracts/"
    notes: str = "Contracts should be published to an artefact repo and validated in CI."

    def validate(self):
        if not isinstance(self.providers, list) or not isinstance(self.consumers, list):
            raise StrategyValidationError("providers and consumers must be lists")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "providers": list(self.providers),
            "consumers": list(self.consumers),
            "tooling": list(self.tooling),
            "publish_location": self.publish_location,
            "notes": self.notes,
        }

    def summary(self) -> str:
        return f"ContractStrategy(required={self.required}, providers={len(self.providers)}, consumers={len(self.consumers)})"
