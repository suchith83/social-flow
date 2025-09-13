# IPA signature / manifest verification helpers
"""
IPA signature & provisioning verification helpers.

Responsibilities:
 - Verify that an IPA is signed (codesign / embedded provisioning) and optionally inspect certs
 - Extract basic metadata (bundle id, version) via `plist` inspection (best-effort)
 - Provide interfaces that call platform tools when available (codesign, security). Falls back to structural checks.

Notes:
 - Production must use Apple's tools and validate provisioning, entitlements, and certificate chains.
"""

import zipfile
import plistlib
import subprocess
import tempfile
from typing import Tuple, Optional, Dict
from .config import CONFIG
import os

class IPASignatureVerifier:
    def __init__(self, require: bool = CONFIG.require_signature_verification):
        self.require = require

    def _has_codesign(self) -> bool:
        try:
            subprocess.run(["codesign", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def inspect_ipa(self, ipa_path: str) -> Dict:
        """
        Extract Info.plist and minimal metadata from IPA.
        """
        if not zipfile.is_zipfile(ipa_path):
            return {}
        with zipfile.ZipFile(ipa_path, "r") as z:
            # find the Info.plist under Payload/*.app/Info.plist
            candidates = [n for n in z.namelist() if n.endswith("Info.plist") and n.count("/") >= 2]
            if not candidates:
                return {}
            info_data = z.read(candidates[0])
            try:
                info = plistlib.loads(info_data)
            except Exception:
                info = {}
            return {
                "plist_path": candidates[0],
                "bundle_id": info.get("CFBundleIdentifier"),
                "version": info.get("CFBundleShortVersionString"),
                "build": info.get("CFBundleVersion"),
            }

    def verify_signature(self, ipa_path: str) -> Tuple[bool, Optional[Dict]]:
        """
        Attempt to verify signature via `codesign` if available.
        Otherwise, perform a light-weight check for presence of _CodeSignature files and embedded provisioning profile.
        """
        if not self.require:
            return True, {"note": "verification disabled"}

        # If codesign available and we can extract the app path to a temp dir, try calling codesign --verify
        if self._has_codesign():
            try:
                tmpdir = tempfile.mkdtemp(prefix="ipa_unpack_")
                with zipfile.ZipFile(ipa_path, "r") as z:
                    z.extractall(tmpdir)
                # find app bundle directory under Payload/*.app
                payload_dir = os.path.join(tmpdir, "Payload")
                apps = [os.path.join(payload_dir, d) for d in os.listdir(payload_dir) if d.endswith(".app")]
                if not apps:
                    return False, {"error": "no_app_bundle_found"}
                app_path = apps[0]
                # call codesign --verify --deep --strict app_path
                subprocess.run(["codesign", "--verify", "--deep", "--strict", app_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # we can also run spctl if needed but skip for demo
                return True, {"note": "codesign_verification_passed"}
            except subprocess.CalledProcessError as e:
                return False, {"error": "codesign_failed", "detail": str(e)}
            except Exception as e:
                return False, {"error": "verification_error", "detail": str(e)}
        else:
            # fallback: check for presence of _CodeSignature directory and embedded.mobileprovision
            try:
                with zipfile.ZipFile(ipa_path, "r") as z:
                    namelist = z.namelist()
                    has_code_sig = any("_CodeSignature" in n for n in namelist)
                    has_provision = any(n.endswith("embedded.mobileprovision") for n in namelist)
                    ok = has_code_sig and has_provision
                    return ok, {"has_code_signature": has_code_sig, "has_provision": has_provision}
            except Exception as e:
                return False, {"error": str(e)}
