"""
Test plan generator and model.

- Accepts a target (feature/PR/release), risk assessment, and desired strategies
- Produces a structured TestPlan that can be serialized and used to drive CI
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from .utils import safe_dumps
from .risk_assessment import RiskLevel, RiskAssessment


@dataclass
class TestCaseEntry:
    id: str
    title: str
    level: str  # unit/integration/e2e/contract
    estimated_minutes: int = 1
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    automated: bool = True
    notes: str = ""


@dataclass
class TestPlan:
    target: str  # e.g., PR number, feature flag, release id
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    test_cases: List[TestCaseEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk: Optional[RiskAssessment] = None

    def add_case(self, case: TestCaseEntry):
        self.test_cases.append(case)

    def total_estimated_time(self) -> int:
        return sum(c.estimated_minutes for c in self.test_cases)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "created_at": self.created_at,
            "test_cases": [c.__dict__ for c in self.test_cases],
            "metadata": self.metadata,
            "risk": self.risk.to_dict() if self.risk else None,
            "total_estimated_minutes": self.total_estimated_time(),
        }

    def to_json(self, pretty: bool = False) -> str:
        return safe_dumps(self.to_dict(), pretty=pretty)

    def summary(self) -> str:
        counts = {}
        for c in self.test_cases:
            counts[c.level] = counts.get(c.level, 0) + 1
        return f"TestPlan(target={self.target}, cases={len(self.test_cases)}, breakdown={counts})"
