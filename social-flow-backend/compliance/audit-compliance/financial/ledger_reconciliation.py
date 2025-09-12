# ledger_reconciliation.py
from collections import defaultdict
from .utils import log_event

class LedgerReconciliation:
    """
    Reconciles multiple ledgers (e.g., bank vs. internal books).
    """

    def reconcile(self, ledger_a, ledger_b):
        """
        Compare two ledgers and return mismatches.
        """
        mismatches = []
        ledger_a_map = defaultdict(float)
        ledger_b_map = defaultdict(float)

        for t in ledger_a:
            ledger_a_map[t['id']] += t['amount']
        for t in ledger_b:
            ledger_b_map[t['id']] += t['amount']

        all_ids = set(ledger_a_map.keys()) | set(ledger_b_map.keys())
        for tid in all_ids:
            if round(ledger_a_map[tid], 2) != round(ledger_b_map[tid], 2):
                mismatches.append({
                    "transaction_id": tid,
                    "ledger_a": ledger_a_map[tid],
                    "ledger_b": ledger_b_map[tid]
                })

        if mismatches:
            log_event(f"Reconciliation mismatches: {mismatches}", "WARNING")
        else:
            log_event("Reconciliation successful")
        return mismatches
