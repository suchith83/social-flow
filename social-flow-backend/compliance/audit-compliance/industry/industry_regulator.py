# industry_regulator.py
from .hipaa_checker import HIPAAChecker
from .pci_checker import PCIChecker
from .gdpr_checker import GDPRChecker
from .basel_checker import BaselChecker
from .utils import log_event

class IndustryRegulator:
    """
    Central dispatcher for industry compliance checks.
    """

    def __init__(self):
        self.checkers = {
            "HIPAA": HIPAAChecker(),
            "PCI": PCIChecker(),
            "GDPR": GDPRChecker(),
            "Basel": BaselChecker(),
        }

    def run_checks(self, industry, dataset):
        if industry not in self.checkers:
            raise ValueError(f"Unsupported industry: {industry}")
        checker = self.checkers[industry]
        findings = checker.check(dataset)
        if not findings:
            log_event(f"{industry} compliance passed ✅")
        return findings
