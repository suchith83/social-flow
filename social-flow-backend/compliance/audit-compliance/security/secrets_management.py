# secrets_management.py
import os
import base64
import json
from typing import Dict, Optional
from .utils import log_event, secure_hash, timing_safe_compare

# This module intentionally keeps a pluggable adapter design:
# - In production, adapter can be HashiCorp Vault, AWS Secrets Manager, Azure KeyVault etc.
# - Here we provide a simple file-backed adapter for local testing but with secure defaults.

class SecretsAdapter:
    """Abstract base adapter interface."""
    def get_secret(self, name: str) -> Optional[str]:
        raise NotImplementedError()
    def set_secret(self, name: str, value: str):
        raise NotImplementedError()
    def rotate_secret(self, name: str, rotation_fn):
        raise NotImplementedError()

class FileSecretsAdapter(SecretsAdapter):
    """
    Simple file-based secret store.
    - Stores base64-encoded JSON entries in a single file with minimal obfuscation.
    - Not for production — only as a pluggable adapter for dev/test.
    """
    def __init__(self, path: str = "secrets.json", encryption_key: Optional[str] = None):
        self.path = path
        self.encryption_key = encryption_key or os.environ.get("SECRETS_KEY", None)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_store(self) -> Dict[str, str]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_store(self, store: Dict[str, str]):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)

    def _encrypt(self, plaintext: str) -> str:
        # minimal pseudo-encryption using HMAC/salt then base64 (not secure for prod)
        salt = secure_hash(plaintext)[:8]
        payload = f"{salt}:{plaintext}"
        return base64.b64encode(payload.encode("utf-8")).decode("utf-8")

    def _decrypt(self, ciphertext: str) -> str:
        decoded = base64.b64decode(ciphertext.encode("utf-8")).decode("utf-8")
        # return payload after salt:
        return decoded.split(":", 1)[1] if ":" in decoded else decoded

    def get_secret(self, name: str) -> Optional[str]:
        store = self._load_store()
        val = store.get(name)
        if not val:
            log_event(f"Secret not found: {name}", level="WARNING")
            return None
        try:
            secret = self._decrypt(val)
            log_event(f"Secret retrieved: {name}", level="INFO")
            return secret
        except Exception as e:
            log_event(f"Failed to decrypt secret {name}: {e}", level="ERROR")
            return None

    def set_secret(self, name: str, value: str):
        store = self._load_store()
        store[name] = self._encrypt(value)
        self._save_store(store)
        log_event(f"Secret stored: {name}", level="INFO")

    def rotate_secret(self, name: str, rotation_fn):
        """
        rotation_fn takes old_secret (or None) and returns new_secret.
        """
        old = self.get_secret(name)
        new = rotation_fn(old)
        if new is None:
            log_event(f"Rotation function returned None for {name}, abort.", level="ERROR")
            return False
        self.set_secret(name, new)
        log_event(f"Secret rotated: {name}", level="INFO")
        return True

class SecretsManager:
    """
    Manager wrapping an adapter and offering discovery/scan helpers.
    """
    def __init__(self, adapter: SecretsAdapter):
        self.adapter = adapter

    def discover_in_files(self, root_path: str, patterns=("*.env", "*.yaml", "*.yml", "*.json")) -> Dict[str, List[str]]:
        """
        Heuristic scan that looks for likely secrets in files by patterns and regex heuristics.
        Returns mapping: filename -> [matches]
        """
        import os, fnmatch, re
        secret_regex = re.compile(r"(?:api[_-]?key|secret|token|passwd|password)[\"'\s:=]+([A-Za-z0-9\-_+=/]{8,})", re.I)
        findings = {}
        for base, dirs, files in os.walk(root_path):
            for p in patterns:
                for fname in fnmatch.filter(files, p):
                    path = os.path.join(base, fname)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        matches = secret_regex.findall(content)
                        if matches:
                            findings[path] = matches
                            log_event(f"Secret-like tokens discovered in {path}", level="WARNING", file=path, count=len(matches))
                    except Exception as e:
                        continue
        return findings
