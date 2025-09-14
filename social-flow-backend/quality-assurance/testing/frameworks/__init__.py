"""
quality_assurance.testing.frameworks

Unified API and helpers for working with test frameworks (pytest, unittest).
Provides adapters, runners, reporters and CI hooks to run tests programmatically.
"""

__version__ = "1.0.0"

from .config import FRAMEWORKS_CONFIG
from .runner import TestRunner, run_tests_cli
from .adapters import PytestAdapter, UnittestAdapter
from .reporter import JSONReporter, SummaryReporter
from .plugins import pytest_register_plugins
from .ci_integration import CIIntegration
