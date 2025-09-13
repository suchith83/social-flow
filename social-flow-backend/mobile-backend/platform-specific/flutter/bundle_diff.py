# Create/serve Flutter bundle diffs (delta updates)
"""
Flutter bundle & AOT diffing utilities.

Responsibilities:
 - Produce diffs between Flutter engine/app bundles (Android .aab split, iOS .ipa, or raw flutter_asset bundles)
 - Decide best package to serve for a client (full bundle vs patch)
 - Provide apply-diff facility (demo naive algorithm). Replace with bsdiff/courgette/Google delta tools in prod.

Notes:
 - Flutter update systems vary: for simple asset updates you may provide resource patches; for code (AOT) updating is platform-limited.
 - For Android, consider using 'playstore' delta mechanisms; for iOS, use app thinning & on-demand resources; for in-app updates consider code push via Dart/Flutter tools (if allowed).
"""

import os
import hashlib
from typing import Optional, Tuple
from .config import CONFIG
from .utils import ensure_dir, read_file, store_file, file_size

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

class FlutterBundleDiffer:
    def __init__(self, bundle_dir: str = CONFIG.bundle_storage_dir, diffs_dir: str = CONFIG.diffs_storage_dir):
        self.bundle_dir = bundle_dir
        self.diffs_dir = diffs_dir
        ensure_dir(self.bundle_dir)
        ensure_dir(self.diffs_dir)

    def _diff_path(self, from_sha: str, to_sha: str) -> str:
        name = f"diff_{from_sha[:12]}_to_{to_sha[:12]}.bin"
        return os.path.join(self.diffs_dir, name)

    def produce_diff(self, from_path: str, to_path: str) -> Tuple[Optional[str], dict]:
        if not CONFIG.enable_bundle_diffing:
            return None, {"reason": "diffing_disabled"}
        if not os.path.exists(from_path) or not os.path.exists(to_path):
            return None, {"reason": "missing_files"}

        size_from = file_size(from_path)
        size_to = file_size(to_path)
        if min(size_from, size_to) < CONFIG.min_diff_size_bytes:
            return None, {"reason": "too_small", "size_from": size_from, "size_to": size_to}

        # naive XOR diff for demo only
        a = read_file(from_path) or b""
        b = read_file(to_path) or b""
        n = max(len(a), len(b))
        diff = bytearray(n)
        for i in range(n):
            ai = a[i] if i < len(a) else 0
            bi = b[i] if i < len(b) else 0
            diff[i] = ai ^ bi

        fp_from = _sha256(a)
        fp_to = _sha256(b)
        diff_path = self._diff_path(fp_from, fp_to)
        store_file(diff_path, bytes(diff))
        meta = {"from_size": len(a), "to_size": len(b), "diff_bytes": len(diff), "sha_from": fp_from, "sha_to": fp_to}
        return diff_path, meta

    def apply_diff(self, from_path: str, diff_path: str, out_path: str) -> bool:
        if not os.path.exists(from_path) or not os.path.exists(diff_path):
            return False
        a = read_file(from_path) or b""
        diff = read_file(diff_path) or b""
        n = max(len(a), len(diff))
        out = bytearray(n)
        for i in range(n):
            ai = a[i] if i < len(a) else 0
            di = diff[i] if i < len(diff) else 0
            out[i] = ai ^ di
        store_file(out_path, bytes(out))
        return True

    def best_package_for_client(self, client_meta: dict, available_bundles: dict) -> dict:
        """
        Decide whether to give full bundle or a diff.
        client_meta e.g. {"installed_sha": "...", "platform": "android", "abi": "arm64"}
        available_bundles e.g. {"v1.2.3": {"path": "...", "sha": "..."}, ...}
        """
        if not available_bundles:
            return {"type": "none", "reason": "no_bundles"}
        # pick latest by version key lexicographically (replace with semver-aware ordering in prod)
        latest = sorted(available_bundles.items(), reverse=True)[0][1]
        installed_sha = client_meta.get("installed_sha")
        if not installed_sha:
            return {"type": "full", "path": latest["path"], "meta": {"reason": "no_installed"}}
        matched = None
        for v, info in available_bundles.items():
            if info.get("sha") == installed_sha:
                matched = info
                break
        if not matched:
            return {"type": "full", "path": latest["path"], "meta": {"reason": "installed_not_found"}}
        diff_path, meta = self.produce_diff(matched["path"], latest["path"])
        if diff_path:
            return {"type": "diff", "path": diff_path, "meta": meta}
        else:
            return {"type": "full", "path": latest["path"], "meta": {"reason": "diff_unavailable"}}
