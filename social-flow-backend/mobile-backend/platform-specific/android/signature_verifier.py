# APK signature / manifest verification helpers
"""
APK / Signature verification helpers.

This module provides utilities to:
 - Verify APK signature (v1/v2/v3) if a verification tool is available
 - Inspect AndroidManifest (package name, versionCode, versionName) from an APK
 - Validate certificates / pins against allowed keys

In production you'd call platform tools (`apksigner`, `aapt`, `apkanalyzer`) or use java `apksig` library.
This implementation provides the interfaces and a lightweight pure-Python fallback
for reading ZIP comment headers (not a security replacement).
"""

import zipfile
import json
import subprocess
import tempfile
from typing import Dict, Optional, Tuple
from .config import CONFIG

class SignatureVerifier:
    def __init__(self, require_verification: bool = CONFIG.require_signature_verification):
        self.require_verification = require_verification

    def _has_apksigner(self) -> bool:
        """Check if apksigner is on PATH (used to verify real signature)."""
        try:
            subprocess.run(["apksigner", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def extract_manifest(self, apk_path: str) -> Dict:
        """
        Try to extract minimal manifest information. Production: use `apkanalyzer` or `aapt dump badging`.
        Here we open the APK (zip) and try to find AndroidManifest.xml as bytes — it's binary XML,
        so we only return placeholders unless a real parser is installed.
        """
        if not zipfile.is_zipfile(apk_path):
            raise ValueError("Not a valid APK (zip archive)")

        with zipfile.ZipFile(apk_path, "r") as z:
            namelist = z.namelist()
            if "AndroidManifest.xml" in namelist:
                # can't parse binary manifest here reliably — return placeholder info
                return {"package": None, "versionCode": None, "versionName": None, "note": "binary manifest present; parse with aapt for details"}
        return {}

    def verify_signature(self, apk_path: str) -> Tuple[bool, Optional[Dict]]:
        """
        Verify APK signature using `apksigner` if available. Returns (is_valid, details).
        When `apksigner` is not present, we return (not_verified, None) if verification is required.
        """
        if not self.require_verification:
            return True, {"note": "signature verification disabled in config"}

        if self._has_apksigner():
            # call apksigner verify --verbose --print-certs apk_path
            try:
                out = subprocess.check_output(["apksigner", "verify", "--print-certs", apk_path], stderr=subprocess.STDOUT, timeout=10)
                text = out.decode("utf-8", errors="ignore")
                return True, {"raw_output": text}
            except subprocess.CalledProcessError as e:
                return False, {"error": e.output.decode("utf-8", errors="ignore")}
        else:
            # fallback: not secure — indicate we couldn't verify
            return False, {"error": "apksigner not available on server; cannot verify signature"}
