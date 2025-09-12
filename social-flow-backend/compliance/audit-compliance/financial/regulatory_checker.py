# regulatory_checker.py
from .utils import log_event

class RegulatoryChecker:
    """
    Maps financial data against regulations (SOX, IFRS, GAAP).
    """

    def __init__(self):
        self.rules = {
            "SOX": ["audit trail required", "segregation of duties"],
            "IFRS": ["fair value reporting", "impairment review"],
            "GAAP": ["accrual basis", "consistency principle"]
        }

    def check_compliance(self, transactions, regulation="SOX"):
        """
        Apply basic compliance rules (mock implementation).
        """
        findings = []
        if regulation not in self.rules:
            raise ValueError(f"Unsupported regulation: {regulation}")

        rules = self.rules[regulation]
        for r in rules:
            # Mock compliance check
            if len(transactions) < 1:  # Example rule trigger
                findings.append(f"Violation: {r}")
        
        if findings:
            log_event(f"Compliance violations found: {findings}", "ALERT")
        else:
            log_event(f"All {regulation} checks passed")
        return findings
