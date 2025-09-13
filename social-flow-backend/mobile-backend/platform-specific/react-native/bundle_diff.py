# Create/serve React Native bundle diffs (delta updates)
"""
JS Bundle / OTA diffing for React Native.

Implements a CodePush-like interface:
 - store JS bundles (full)
 - optionally produce delta updates between bundle versions
 - decide best package (full vs delta) for a client
 - naive diff algorithm for demo only (XOR) — replace with bsdiff/rsync/Google tools in prod
 - apply-delta function to reconstruct

Important production notes:
 - Real OTA updates require careful signature checks, metadata, rollout strategy, and client support.
 - Ensure compatibility between JS bytecode and native modules / assets.
"""

import os
import hashlib
from typing import Optional, Tuple, Dict
from .config import CONFIG
from .utils import ensure_dir, read_file, store_file, file_size, safe_key, sha256_hex

class RNBundleDiffer:
    def __init__(self, bundle_dir: str = CONFIG.bundle_storage_dir, diffs_dir: str = CONFIG.diffs_storage_dir):
        self.bundle_dir = bundle_dir
        self.diffs_dir = diffs_dir
        ensure_dir(self.bundle_dir)
        ensure_dir(self.diffs_dir)

    def _bundle_path(self, name: str) -> str:
        return os.path.join(self.bundle_dir, safe_key(name))

    def _diff_path(self, from_sha: str, to_sha: str) -> str:
        name = f"diff_{from_sha[:12]}_to_{to_sha[:12]}.bin"
        return os.path.join(self.diffs_dir, name)

    def store_bundle(self, name: str, data: bytes) -> Dict:
        """
        Store a full JS bundle and return metadata (path, sha, size).
        name: logical name, e.g., "android-arm64_v1.2.3"
        """
        path = self._bundle_path(name)
        store_file(path, data)
        return {"path": path, "sha": sha256_hex(data), "size": len(data)}

    def produce_diff(self, from_path: str, to_path: str) -> Tuple[Optional[str], dict]:
        """
        Produce diff between two bundle files. Returns diff_path and metadata.
        Uses naive XOR diff for demo; replace with better algorithm in prod.
        """
        if not CONFIG.enable_bundle_diffing:
            return None, {"reason": "diffing_disabled"}
        if not os.path.exists(from_path) or not os.path.exists(to_path):
            return None, {"reason": "missing_file"}
        a = read_file(from_path) or b""
        b = read_file(to_path) or b""
        size_from = len(a)
        size_to = len(b)
        if min(size_from, size_to) < CONFIG.min_diff_size_bytes:
            return None, {"reason": "too_small", "size_from": size_from, "size_to": size_to}
        n = max(size_from, size_to)
        diff = bytearray(n)
        for i in range(n):
            ai = a[i] if i < size_from else 0
            bi = b[i] if i < size_to else 0
            diff[i] = ai ^ bi
        diff_path = self._diff_path(sha256_hex(a), sha256_hex(b))
        store_file(diff_path, bytes(diff))
        # only use diff if it's smaller than configured ratio
        if file_size(diff_path) > CONFIG.max_delta_size_ratio * size_to:
            # delete diff if not useful
            os.remove(diff_path)
            return None, {"reason": "diff_larger_than_threshold", "diff_bytes": file_size(diff_path), "to_size": size_to}
        meta = {"from_size": size_from, "to_size": size_to, "diff_bytes": file_size(diff_path), "sha_from": sha256_hex(a), "sha_to": sha256_hex(b)}
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

    def best_package_for_client(self, client_meta: dict, available_bundles: Dict[str, dict]) -> dict:
        """
        Decide full bundle or diff for client.
        client_meta contains installed bundle sha, platform/arch.
        available_bundles: {"v1.2.3": {"path": "...", "sha": "...", "name": "..."}}
        """
        if not available_bundles:
            return {"type": "none", "reason": "no_bundles"}
        # pick latest (simple lexicographic; replace with semver)
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
        return {"type": "full", "path": latest["path"], "meta": {"reason": "diff_unavailable"}}
