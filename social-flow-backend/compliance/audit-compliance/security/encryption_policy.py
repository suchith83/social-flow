# encryption_policy.py
from typing import Dict, Optional
import os
from .utils import log_event, secure_hash

class EncryptionPolicy:
    """
    Policy enforcement helper for encryption-at-rest and in-transit rules.
    Designed to be used by auditors to validate asset metadata against policy.
    """

    DEFAULT_POLICY = {
        "min_key_length_bits": 256,
        "approved_algorithms": ["AES-GCM", "AES-CBC", "RSA-OAEP"],
        "key_rotation_days": 365,
        "disk_encryption_required": True,
        "transport_tls_min_version": "1.2"
    }

    def __init__(self, policy: Optional[Dict] = None):
        self.policy = policy or self.DEFAULT_POLICY

    def evaluate_asset(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single asset metadata against the encryption policy.
        metadata example:
        {
            "asset_id": "db-1",
            "disk_encrypted": True,
            "cipher": "AES-GCM",
            "key_length": 256,
            "key_last_rotation": "2025-01-01",
            "tls_min_version": "1.2"
        }
        Returns dict with findings and boolean 'compliant'
        """
        findings = []
        compliant = True
        if self.policy.get("disk_encryption_required", True) and not metadata.get("disk_encrypted", False):
            findings.append("Disk not encrypted")
            compliant = False
        if metadata.get("cipher") not in self.policy.get("approved_algorithms", []):
            findings.append(f"Unapproved cipher: {metadata.get('cipher')}")
            compliant = False
        if metadata.get("key_length", 0) < self.policy.get("min_key_length_bits", 256):
            findings.append(f"Insufficient key length: {metadata.get('key_length')}")
            compliant = False
        # check rotation
        from datetime import datetime
        try:
            last_rot = datetime.fromisoformat(metadata.get("key_last_rotation"))
            age_days = (datetime.utcnow() - last_rot).days
            if age_days > self.policy.get("key_rotation_days", 365):
                findings.append(f"Key rotation overdue by {age_days - self.policy.get('key_rotation_days',365)} days")
                compliant = False
        except Exception:
            # if missing or malformed date, mark as non-compliant
            findings.append("Missing or malformed key_last_rotation")
            compliant = False
        # TLS version check
        if metadata.get("tls_min_version", "1.0") < self.policy.get("transport_tls_min_version", "1.2"):
            findings.append(f"TLS version too low: {metadata.get('tls_min_version')}")
            compliant = False
        result = {"asset_id": metadata.get("asset_id"), "compliant": compliant, "findings": findings}
        if not compliant:
            log_event(f"Encryption policy non-compliant: {result}", level="WARNING", asset=metadata.get("asset_id"))
        else:
            log_event(f"Asset {metadata.get('asset_id')} is encryption-compliant", level="INFO")
        return result

    def generate_policy_fingerprint(self) -> str:
        """Return a hash fingerprint for the current policy for audit traceability."""
        import json
        canonical = json.dumps(self.policy, sort_keys=True, separators=(",", ":"))
        return secure_hash(canonical, salt="encryption-policy")
