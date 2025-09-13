# Bundle signature / manifest verification helpers
"""
Signature / package verification for Flutter bundles.

Depending on platform:
 - Android: verify AAB/APK signatures using apksigner or Play Integrity if available
 - iOS: verify IPA signatures / embedded provisioning profiles (needs codesign / security tools)
 - For raw Flutter bundles we may verify checksums and expected engine signatures.

This module provides interfaces and basic fallback behavior.
"""

import zipfile
import subprocess
from typing import Tuple, Optional
from .config import CONFIG

class FlutterSignatureVerifier:
    def __init__(self, require: bool = CONFIG.require_signature_verification):
        self.require = require

    def verify_android_apk(self, apk_path: str) -> Tuple[bool, Optional[dict]]:
        if not self.require:
            return True, {"note": "verification disabled"}

        # attempt to run apksigner if available
        try:
            subprocess.run(["apksigner", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            out = subprocess.check_output(["apksigner", "verify", "--print-certs", apk_path], stderr=subprocess.STDOUT, timeout=15)
            return True, {"raw": out.decode("utf-8", errors="ignore")}
        except Exception as e:
            return False, {"error": str(e)}

    def verify_ios_ipa(self, ipa_path: str) -> Tuple[bool, Optional[dict]]:
        if not self.require:
            return True, {"note": "verification disabled"}
        # In production, use codesign / security tools to validate. Here we just check zip integrity.
        try:
            if not zipfile.is_zipfile(ipa_path):
                return False, {"error": "not_zip"}
            with zipfile.ZipFile(ipa_path, "r") as z:
                # check for embedded mobile provisioning or code signature files (best-effort)
                namelist = z.namelist()
                has_signature = any(n.startswith("Payload/") and (n.endswith(".app/_CodeSignature/") or "_CodeSignature" in n) for n in namelist)
                return has_signature, {"has_signature": has_signature}
        except Exception as e:
            return False, {"error": str(e)}
