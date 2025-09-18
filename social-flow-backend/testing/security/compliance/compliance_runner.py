# =========================
# File: testing/security/compliance/compliance_runner.py
# =========================
"""
Main runner that loads policies, executes all compliance checks,
and generates reports.
"""

import os
import yaml
import json
from pathlib import Path
from .policy_engine import PolicyEngine
from .standards import gdpr_checker, hipaa_checker, soc2_checker
from .reports import report_generator
from .utils import logger

class ComplianceRunner:
    def __init__(self, policy_file="compliance_policies.yaml"):
        self.policy_file = Path(policy_file)
        self.log = logger.get_logger("ComplianceRunner")
        self.policy_engine = None

    def load_policies(self):
        if not self.policy_file.exists():
            raise FileNotFoundError(f"Policy file not found: {self.policy_file}")
        with open(self.policy_file, "r", encoding="utf-8") as f:
            policies = yaml.safe_load(f)
        self.policy_engine = PolicyEngine(policies)
        self.log.info(f"Loaded {len(policies.get('policies', []))} compliance policies")

    def run_checks(self):
        if not self.policy_engine:
            self.load_policies()

        self.log.info("Running compliance checks across GDPR, HIPAA, SOC2")
        results = {
            "gdpr": gdpr_checker.run_checks(self.policy_engine),
            "hipaa": hipaa_checker.run_checks(self.policy_engine),
            "soc2": soc2_checker.run_checks(self.policy_engine),
        }
        return results

    def generate_report(self, results, fmt="json"):
        report_path = Path("compliance_report." + fmt)
        report_generator.generate(results, report_path, fmt)
        self.log.info(f"Compliance report generated: {report_path}")
        return report_path

if __name__ == "__main__":
    runner = ComplianceRunner()
    runner.load_policies()
    results = runner.run_checks()
    runner.generate_report(results, fmt="json")
