"""
CI policy definitions and enforcement.

- Translate strategies and risk assessments into CI gating rules
- Provide programmatic enforcement helpers that can be used in CI scripts
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import os
from .config import STRATEGIES_CONFIG
from .exceptions import EnforcementError
from .utils import safe_dumps


@dataclass
class EnforcementResult:
    success: bool
    messages: List[str]


class CIPolicy:
    """
    Encapsulates CI gating logic derived from strategies configuration and runtime signals.
    """

    def __init__(self, config=STRATEGIES_CONFIG):
        self.config = config

    def required_checks(self, env: str = "ci") -> List[str]:
        """
        Determine which checks are required in the given environment.
        """
        matrix = self.config.default_matrix
        return matrix.get(env, [])

    def evaluate_coverage(self, metrics: Dict[str, float]) -> EnforcementResult:
        """
        Evaluate coverage metrics (expects dict like {"unit": 92.3, "integration": 81})
        against thresholds in config. Returns EnforcementResult.
        """
        msgs = []
        failed = False
        unit_thresh = self.config.default_unit_coverage_threshold
        int_thresh = self.config.default_integration_coverage_threshold

        unit_cov = metrics.get("unit")
        if unit_cov is None:
            msgs.append("unit coverage metric missing")
            failed = True
        else:
            msgs.append(f"unit coverage: {unit_cov:.2f}% (required: {unit_thresh:.2f}%)")
            if unit_cov < unit_thresh:
                msgs.append("unit coverage threshold not met")
                failed = True

        int_cov = metrics.get("integration")
        if int_cov is not None:
            msgs.append(f"integration coverage: {int_cov:.2f}% (required: {int_thresh:.2f}%)")
            if int_cov < int_thresh:
                msgs.append("integration coverage threshold not met")
                failed = True
        else:
            msgs.append("integration coverage metric missing; skipping integration threshold check")

        return EnforcementResult(success=not failed, messages=msgs)

    def enforce(self, metrics: Dict[str, float], risk_assessment: Dict[str, Any] = None) -> EnforcementResult:
        """
        Top-level enforcement combining coverage and risk-based gating.
        Returns EnforcementResult; callers should act on success/failure.
        """
        msgs = []
        # Evaluate coverage
        cov_result = self.evaluate_coverage(metrics)
        msgs.extend(cov_result.messages)
        if not cov_result.success:
            return EnforcementResult(success=False, messages=msgs)

        # Optionally apply risk rules
        if risk_assessment:
            lvl = risk_assessment.get("level")
            if lvl == "high" and self.config.max_risk_level != "high":
                msgs.append(f"High risk change detected (score {risk_assessment.get('score')}). Blocking merge per config.")
                return EnforcementResult(success=False, messages=msgs)
            msgs.append(f"Risk level {lvl} allowed by policy.")
        else:
            msgs.append("No risk assessment provided; defaulting to policy pass for risk.")

        msgs.append("All policy checks passed.")
        return EnforcementResult(success=True, messages=msgs)

    def to_json(self, obj: Any, pretty: bool = False) -> str:
        return safe_dumps(obj, pretty=pretty)
