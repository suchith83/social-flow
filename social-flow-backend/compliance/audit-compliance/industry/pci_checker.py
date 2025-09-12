# pci_checker.py
from .utils import log_event, secure_hash

class PCIChecker:
    """
    Enforces PCI-DSS rules for payment processing.
    """

    def __init__(self):
        self.rules = [
            "Card data must not be stored in plaintext",
            "Strong encryption required (AES-256+)",
            "Mask PAN (Primary Account Number)"
        ]

    def check(self, transactions):
        findings = []
        for tx in transactions:
            if "card_number" in tx and len(tx["card_number"]) > 6:
                if tx.get("plaintext_storage", False):
                    findings.append("Card stored in plaintext")
                if not tx.get("encrypted", False):
                    findings.append("Unencrypted card transaction")
        if findings:
            log_event(f"PCI violations: {findings}", "ALERT")
        return findings
