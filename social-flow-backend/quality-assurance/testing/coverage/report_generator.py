"""
Report generator - aggregate and present results in various formats.
"""

import json
import os
from datetime import datetime
from .config import CONFIG


class ReportGenerator:
    """Generate detailed JSON and markdown summary reports."""

    def __init__(self, config=CONFIG):
        self.config = config

    def generate_summary(self):
        """Generate a Markdown summary report."""
        json_path = os.path.join(self.config.report_dir, "coverage.json")
        if not os.path.exists(json_path):
            return "# Coverage Report\n\nNo coverage.json found.\n"

        with open(json_path, "r") as f:
            data = json.load(f)

        total = data["totals"]
        md = [
            "# 📊 Coverage Report",
            f"Generated on: {datetime.now().isoformat()}",
            "",
            f"- Total statements: {total['num_statements']}",
            f"- Missing: {total['missing_lines']}",
            f"- Excluded: {total['excluded_lines']}",
            f"- Coverage: **{total['percent_covered']}%**",
        ]
        return "\n".join(md)

    def save_summary(self):
        """Save markdown summary to file."""
        path = os.path.join(self.config.report_dir, "SUMMARY.md")
        with open(path, "w") as f:
            f.write(self.generate_summary())
        return path
