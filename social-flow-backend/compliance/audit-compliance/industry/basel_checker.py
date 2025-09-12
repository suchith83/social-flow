# basel_checker.py
from .utils import log_event

class BaselChecker:
    """
    Enforces Basel III banking compliance.
    - Capital adequacy
    - Liquidity coverage ratio (LCR)
    """

    def __init__(self):
        self.minimum_capital_ratio = 0.08  # 8%
        self.minimum_lcr = 1.0  # 100%

    def check(self, bank_metrics):
        findings = []
        if bank_metrics["capital_ratio"] < self.minimum_capital_ratio:
            findings.append(f"Capital ratio too low: {bank_metrics['capital_ratio']}")
        if bank_metrics["lcr"] < self.minimum_lcr:
            findings.append(f"LCR too low: {bank_metrics['lcr']}")
        if findings:
            log_event(f"Basel III violations: {findings}", "ERROR")
        return findings
