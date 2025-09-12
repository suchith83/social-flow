# tests/test_industry.py
import unittest
from compliance.audit_compliance.industry.hipaa_checker import HIPAAChecker
from compliance.audit_compliance.industry.pci_checker import PCIChecker
from compliance.audit_compliance.industry.gdpr_checker import GDPRChecker
from compliance.audit_compliance.industry.basel_checker import BaselChecker

class TestIndustryCompliance(unittest.TestCase):

    def test_hipaa(self):
        checker = HIPAAChecker()
        records = [{"patient_id": "12345", "encrypted": False, "access_log": False}]
        findings = checker.check(records)
        self.assertGreater(len(findings), 0)

    def test_pci(self):
        checker = PCIChecker()
        txs = [{"card_number": "4111111111111111", "plaintext_storage": True}]
        findings = checker.check(txs)
        self.assertGreater(len(findings), 0)

    def test_gdpr(self):
        checker = GDPRChecker()
        users = [{"id": "u1", "consent": False}]
        findings = checker.check(users)
        self.assertIn("No consent", findings[0])

    def test_basel(self):
        checker = BaselChecker()
        metrics = {"capital_ratio": 0.05, "lcr": 0.8}
        findings = checker.check(metrics)
        self.assertGreater(len(findings), 0)

if __name__ == "__main__":
    unittest.main()
