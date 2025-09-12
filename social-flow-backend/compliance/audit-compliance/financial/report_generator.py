# report_generator.py
import json
import csv
from datetime import datetime
from .utils import log_event

class ReportGenerator:
    """
    Generates compliance reports in multiple formats.
    """

    def __init__(self, company_name):
        self.company = company_name

    def generate_json_report(self, findings, filepath):
        data = {
            "company": self.company,
            "timestamp": datetime.utcnow().isoformat(),
            "findings": findings
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        log_event(f"JSON report generated at {filepath}")

    def generate_csv_report(self, findings, filepath):
        keys = findings[0].keys() if findings else []
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(findings)
        log_event(f"CSV report generated at {filepath}")

    def generate_text_summary(self, findings):
        """
        Human-readable summary.
        """
        lines = [f"Compliance Report for {self.company}"]
        lines.append("="*50)
        for f in findings:
            lines.append(f"- {f}")
        return "\n".join(lines)
