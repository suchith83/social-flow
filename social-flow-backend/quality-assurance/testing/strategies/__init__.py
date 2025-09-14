"""
quality_assurance.testing.strategies

High-level test strategy definitions, planners, and CI enforcement helpers.
"""

__version__ = "1.0.0"

from .config import STRATEGIES_CONFIG
from .strategy_registry import StrategyRegistry
from .unit_strategy import UnitStrategy
from .integration_strategy import IntegrationStrategy
from .e2e_strategy import E2EStrategy
from .contract_strategy import ContractStrategy
from .test_plan import TestPlan, TestCaseEntry
from .risk_assessment import RiskAssessment, RiskLevel
from .ci_policy import CIPolicy, EnforcementResult
