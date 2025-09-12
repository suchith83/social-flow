# transaction_audit.py
import hashlib
import statistics
from datetime import datetime
from .utils import secure_hash, log_event

class TransactionAudit:
    """
    Core transaction auditing engine.
    - Validates double-entry bookkeeping.
    - Detects anomalies and fraud heuristics.
    """

    def __init__(self):
        self.audit_log = []

    def validate_double_entry(self, transactions):
        """
        Ensures that debits == credits for every batch of transactions.
        """
        debit_total = sum(t['amount'] for t in transactions if t['type'] == 'debit')
        credit_total = sum(t['amount'] for t in transactions if t['type'] == 'credit')
        
        if round(debit_total, 2) != round(credit_total, 2):
            log_event("Double-entry mismatch detected", "ERROR")
            return False
        return True

    def detect_anomalies(self, transactions):
        """
        Detects statistical anomalies (e.g., outliers).
        """
        amounts = [t['amount'] for t in transactions]
        if not amounts:
            return []
        
        mean = statistics.mean(amounts)
        stdev = statistics.pstdev(amounts)
        
        anomalies = [
            t for t in transactions
            if abs(t['amount'] - mean) > 3 * stdev
        ]
        
        for a in anomalies:
            log_event(f"Anomaly detected: {a}", "WARNING")
        return anomalies

    def fraud_heuristics(self, transactions):
        """
        Apply heuristic checks for fraud:
        - Odd-hour large transactions
        - Repeated amounts just below approval thresholds
        """
        flagged = []
        for t in transactions:
            hour = datetime.fromisoformat(t['timestamp']).hour
            if t['amount'] > 100000 and (hour < 6 or hour > 22):
                flagged.append(t)
            if str(t['amount']).endswith("999"):
                flagged.append(t)

        for f in flagged:
            log_event(f"Fraud heuristic triggered: {f}", "ALERT")
        return flagged
