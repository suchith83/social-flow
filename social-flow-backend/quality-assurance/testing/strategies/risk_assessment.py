"""
Risk assessment helpers.

- Simple risk model with scoring and recommendations.
- Intended as an advisory layer for test planning and CI gating decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class RiskLevel:
    name: str
    score: int  # numeric score, higher means more risky
    description: str = ""


@dataclass
class RiskAssessment:
    """
    Basic risk assessment summarizing impact, likelihood and overall score.
    Score is 0-100 with thresholds mapping to low/medium/high.
    """
    impact: int  # 0-100
    likelihood: int  # 0-100
    vectors: Dict[str, str] = field(default_factory=dict)  # e.g., {"db": "schema change", "auth": "token rotation"}
    rationale: str = ""
    score: int = field(init=False)

    def __post_init__(self):
        # Weighted average: impact heavier than likelihood (60/40)
        self.score = min(100, int((self.impact * 0.6) + (self.likelihood * 0.4)))

    def level(self) -> RiskLevel:
        if self.score >= 75:
            return RiskLevel("high", self.score, "High risk - requires extensive testing and gating")
        elif self.score >= 40:
            return RiskLevel("medium", self.score, "Medium risk - require integration and selective E2E tests")
        else:
            return RiskLevel("low", self.score, "Low risk - unit tests and smoke suffice")

    def recommend(self) -> Dict[str, Any]:
        lvl = self.level()
        if lvl.name == "high":
            return {"run": ["unit", "integration", "e2e", "contract"], "gating": True}
        elif lvl.name == "medium":
            return {"run": ["unit", "integration", "contract"], "gating": True}
        else:
            return {"run": ["unit"], "gating": False}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "impact": self.impact,
            "likelihood": self.likelihood,
            "vectors": dict(self.vectors),
            "rationale": self.rationale,
            "score": self.score,
            "level": self.level().name,
            "recommendation": self.recommend(),
        }
