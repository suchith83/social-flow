# Create/serve IPA diffs (delta updates)
"""
IPA diffing utilities.

Responsibilities:
 - Produce diffs between IPA versions (for delta updates)
 - Serve diff packages with metadata (from -> to)
 - Heuristics: only produce diffs for sufficiently large IPAs
 - Fallback: full IPA if diffing is not beneficial

NOTE:
 - Production diffing should use robust binary diffing tools; this demo uses a naive XOR diff (not suitable for real use).
"""

import os
import hashlib
from typing import Optional, Tuple
from .utils import ensure_dir, read_file, store_file, file_size
from .config import CONFIG

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

class IpaDiffer:
    def __init__(self, storage_dir: str = CONFIG.diffs_storage_dir, ipa_dir: str = CONFIG.ipa_storage_dir):
        self.storage_dir = storage_dir
        self.ipa_dir = ipa_dir
        ensure_dir(self.storage_dir)
        ensure_dir(self.ipa_dir)

    def _diff_path(self, from_sha: str, to_sha: str) -> str:
        safe = f"diff_{from_sha[:12]}_to_{to_sha[:12]}.bin"
        return os.path.join(self.storage_dir, safe)

    def produce_diff(self, from_ipa_path: str, to_ipa_path: str) -> Tuple[Optional[str], dict]:
        if not CONFIG.enable_ipa_diffing:
            return None, {"reason": "diffing_disabled"}
        if not os.path.exists(from_ipa_path) or not os.path.exists(to_ipa_path):
            return None, {"reason": "ipa_missing"}

        size_from = file_size(from_ipa_path)
        size_to = file_size(to_ipa_path)
        if min(size_from, size_to) < CONFIG.min_diff_size_bytes:
            return None, {"reason": "too_small", "size_from": size_from, "size_to": size_to}

        a = read_file(from_ipa_path) or b""
        b = read_file(to_ipa_path) or b""
        n = max(len(a), len(b))
        diff = bytearray(n)
        for i in range(n):
            ai = a[i] if i < len(a) else 0
            bi = b[i] if i < len(b) else 0
            diff[i] = ai ^ bi

        diff_path = self._diff_path(_sha256(a), _sha256(b))
        store_file(diff_path, bytes(diff))
        meta = {"from_size": len(a), "to_size": len(b), "diff_bytes": len(diff), "sha_from": _sha256(a), "sha_to": _sha256(b)}
        return diff_path, meta

    def apply_diff(self, from_ipa_path: str, diff_path: str, output_path: str) -> bool:
        if not os.path.exists(from_ipa_path) or not os.path.exists(diff_path):
            return False
        a = read_file(from_ipa_path) or b""
        diff = read_file(diff_path) or b""
        n = max(len(a), len(diff))
        out = bytearray(n)
        for i in range(n):
            ai = a[i] if i < len(a) else 0
            di = diff[i] if i < len(diff) else 0
            out[i] = ai ^ di
        store_file(output_path, bytes(out))
        return True

    def best_package_for_client(self, client_meta: dict, available_ipas: dict) -> dict:
        """
        Decide whether to give full IPA or diff for a client.
        client_meta e.g. {"installed_sha": "..."}
        available_ipas e.g. {"v1.2.3": {"path":"...", "sha":"..."}, ...}
        """
        if not available_ipas:
            return {"type": "none", "reason": "no_ipas"}
        latest = sorted(available_ipas.items(), reverse=True)[0][1]
        installed = client_meta.get("installed_sha")
        if not installed:
            return {"type": "full", "path": latest["path"], "meta": {"reason": "no_installed"}}
        matched = None
        for v, info in available_ipas.items():
            if info.get("sha") == installed:
                matched = info
                break
        if not matched:
            return {"type": "full", "path": latest["path"], "meta": {"reason": "installed_not_found"}}
        diff_path, meta = self.produce_diff(matched["path"], latest["path"])
        if diff_path:
            return {"type": "diff", "path": diff_path, "meta": meta}
        else:
            return {"type": "full", "path": latest["path"], "meta": {"reason": "diff_unavailable"}}
