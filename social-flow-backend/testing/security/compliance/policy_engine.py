# =========================
# File: testing/security/compliance/policy_engine.py
# =========================
"""
Policy Engine loads compliance policies and provides validation utilities.
"""

import re

class PolicyEngine:
    def __init__(self, policies):
        self.policies = policies.get("policies", [])

    def get_policies_for(self, category):
        return [p for p in self.policies if category in p.get("applies_to", [])]

    def validate(self, category, evidence):
        """
        Validates evidence against policies for a given category.
        Evidence is a dict containing metadata, logs, or API responses.
        """
        results = []
        for policy in self.get_policies_for(category):
            passed = self._evaluate(policy, evidence)
            results.append({"policy": policy["id"], "passed": passed, "severity": policy["severity"]})
        return results

    def _evaluate(self, policy, evidence):
        """
        Example evaluation: match keywords in evidence.
        Real-world use would integrate scanners, DB queries, etc.
        """
        for k, v in evidence.items():
            if isinstance(v, str) and re.search(r"error|fail|unauthorized", v, re.IGNORECASE):
                return False
        return True
