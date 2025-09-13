# Create/serve APK diffs (delta updates)
"""
APK diffing and delta updates.

Responsibilities:
 - Produce binary diffs between APK versions (for OTA/delta updates)
 - Serve diff packages with metadata (from -> to)
 - Heuristics: only produce diffs for sufficiently large APKs
 - Fallback: full APK if diffing is not beneficial

NOTE:
 - Real binary diffing should use specialized tools (bsdiff/rdiff/Google's delta generator).
 - This implementation demonstrates interfaces and a naive XOR-based delta (for demo only).
"""

import os
import hashlib
import tempfile
from typing import Optional, Tuple
from .config import CONFIG
from .utils import ensure_dir, read_file, store_file, file_size

DIFF_PREFIX = "diff:"  # for in-memory index if used


def _sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


class ApkDiffer:
    def __init__(self, storage_dir: str = CONFIG.diffs_storage_dir, apk_dir: str = CONFIG.apk_storage_dir):
        self.storage_dir = storage_dir
        self.apk_dir = apk_dir
        ensure_dir(self.storage_dir)
        ensure_dir(self.apk_dir)

    def _diff_path(self, from_ver: str, to_ver: str) -> str:
        safe = f"diff_{from_ver}_to_{to_ver}.bin"
        return os.path.join(self.storage_dir, safe)

    def produce_diff(self, from_apk_path: str, to_apk_path: str) -> Tuple[Optional[str], dict]:
        """
        Produce a delta between two APK files.
        Returns (diff_path or None if not created, metadata)
        Heuristic: only produce diff when both files exist and size > threshold.
        """
        if not CONFIG.enable_apk_diffing:
            return None, {"reason": "diffing disabled"}

        if not os.path.exists(from_apk_path) or not os.path.exists(to_apk_path):
            return None, {"reason": "apk_missing"}

        size_from = file_size(from_apk_path)
        size_to = file_size(to_apk_path)
        if min(size_from, size_to) < CONFIG.min_diff_size_bytes:
            return None, {"reason": "too_small_for_diff", "size_from": size_from, "size_to": size_to}

        # naive XOR-diff (DEMO ONLY) — not efficient or secure; replace with bsdiff in prod
        with open(from_apk_path, "rb") as f:
            a = f.read()
        with open(to_apk_path, "rb") as f:
            b = f.read()

        # build diff of length max(len(a), len(b))
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

    def apply_diff(self, from_apk_path: str, diff_path: str, output_path: str) -> bool:
        """
        Apply XOR-style diff to recreate target APK.
        (Only works for the naive diff above.)
        """
        if not os.path.exists(from_apk_path) or not os.path.exists(diff_path):
            return False
        with open(from_apk_path, "rb") as f:
            a = f.read()
        with open(diff_path, "rb") as f:
            diff = f.read()
        n = max(len(a), len(diff))
        out = bytearray(n)
        for i in range(n):
            ai = a[i] if i < len(a) else 0
            di = diff[i] if i < len(diff) else 0
            out[i] = ai ^ di
        store_file(output_path, bytes(out))
        return True

    def best_package_for_client(self, client_meta: dict, available_apks: dict) -> dict:
        """
        Decide whether to give full APK or diff for a client.
        client_meta e.g. {"installed_sha": "...", "android_version": "13"}
        available_apks e.g. {"v1.2.3": {"path": "...", "sha": "..."}, "v1.2.4": {...}}
        Returns dict with keys: type: "full"|"diff", path, metadata
        """
        target = sorted(available_apks.items(), reverse=True)[0][1]  # pick latest
        target_sha = target.get("sha")
        installed = client_meta.get("installed_sha")
        if not installed:
            return {"type": "full", "path": target["path"], "meta": {"reason": "no_installed_version"}}

        # find matching installed version in available list
        matched = None
        for v, info in available_apks.items():
            if info.get("sha") == installed:
                matched = info
                break
        if not matched:
            return {"type": "full", "path": target["path"], "meta": {"reason": "installed_not_found"}}

        # attempt to provide diff
        diff_path, meta = self.produce_diff(matched["path"], target["path"])
        if diff_path:
            return {"type": "diff", "path": diff_path, "meta": meta}
        else:
            return {"type": "full", "path": target["path"], "meta": {"reason": "diff_unavailable"}}
