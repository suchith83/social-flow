"""
Configuration module for coverage testing.
Defines defaults and provides environment overrides.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageConfig:
    """Immutable configuration for coverage analysis."""

    report_dir: str = os.getenv("COVERAGE_REPORT_DIR", "coverage_reports")
    html_report: bool = True
    json_report: bool = True
    xml_report: bool = True
    badge: bool = True
    fail_under: float = float(os.getenv("COVERAGE_FAIL_UNDER", "80"))
    parallel: bool = True
    source_dirs: tuple = ("src", "app")
    omit_patterns: tuple = ("*/tests/*",)

    @staticmethod
    def from_env() -> "CoverageConfig":
        """Load config from environment variables."""
        return CoverageConfig(
            report_dir=os.getenv("COVERAGE_REPORT_DIR", "coverage_reports"),
            fail_under=float(os.getenv("COVERAGE_FAIL_UNDER", "80")),
        )


CONFIG = CoverageConfig()
