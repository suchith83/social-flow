"""
Coverage Analyzer - parse reports and enforce thresholds.
"""

import json
import os
from .config import CONFIG
from .exceptions import CoverageThresholdError


class CoverageAnalyzer:
    """Analyze coverage reports and enforce thresholds."""

    def __init__(self, config=CONFIG):
        self.config = config

    def analyze(self):
        """Check coverage.json for summary stats."""
        json_path = os.path.join(self.config.report_dir, "coverage.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError("coverage.json not found, run coverage first.")

        with open(json_path, "r") as f:
            data = json.load(f)

        total_percent = data.get("totals", {}).get("percent_covered", 0)
        if total_percent < self.config.fail_under:
            raise CoverageThresholdError(
                f"Coverage {total_percent}% is below threshold {self.config.fail_under}%"
            )
        return total_percent
