# tests/test_financial.py
import unittest
from compliance.audit_compliance.financial.transaction_audit import TransactionAudit
from compliance.audit_compliance.financial.ledger_reconciliation import LedgerReconciliation
from compliance.audit_compliance.financial.risk_assessment import RiskAssessment

class TestFinancialCompliance(unittest.TestCase):

    def test_double_entry(self):
        audit = TransactionAudit()
        transactions = [
            {"id": 1, "type": "debit", "amount": 100, "timestamp": "2025-09-13T10:00:00"},
            {"id": 2, "type": "credit", "amount": 100, "timestamp": "2025-09-13T10:01:00"},
        ]
        self.assertTrue(audit.validate_double_entry(transactions))

    def test_reconciliation(self):
        rec = LedgerReconciliation()
        ledger_a = [{"id": "T1", "amount": 500}]
        ledger_b = [{"id": "T1", "amount": 500}]
        self.assertEqual(rec.reconcile(ledger_a, ledger_b), [])

    def test_risk_var(self):
        risk = RiskAssessment()
        portfolio = [100, 200, 150, 300]
        var = risk.value_at_risk(portfolio, confidence=0.95)
        self.assertIsInstance(var, (int, float))

if __name__ == "__main__":
    unittest.main()
