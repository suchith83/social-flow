# Deduplication logic for storage efficiency
"""
Content Deduplicator

Implements content-addressable storage helpers and metadata management for dedup.
This small module maintains a mapping from fingerprint -> canonical key and
reference counts for safe deletion.

Notes:
 - In production you'd persist these mappings in a transactional metadata store
   (e.g., PostgreSQL, DynamoDB) so that concurrency and crashes are handled.
 - Here we implement an in-memory/small-file-based mapping with careful docstrings
   so it's straightforward to replace.

APIs:
 - ensure_unique(data) -> (fingerprint, already_exists: bool, canonical_key)
 - ref/_unref to manage reference counts
"""

import os
import json
import threading
from typing import Tuple, Optional
from .compression_adapter import CompressionAdapter

class Deduplicator:
    def __init__(self, persistence_path: str = ".dedup_index.json"):
        self.lock = threading.RLock()
        self.index_path = persistence_path
        # index: fingerprint -> {"key": canonical_key, "refs": int, "metadata": {...}}
        self.index = {}
        self._load_index()
        self.compressor = CompressionAdapter()

    def _load_index(self):
        try:
            if os.path.exists(self.index_path):
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self.index = json.load(f)
        except Exception:
            self.index = {}

    def _persist_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def ensure_unique(self, data: bytes, suggested_key: str = None) -> Tuple[str, bool, Optional[str]]:
        """
        Returns fingerprint, already_exists, canonical_key.
        If already exists, canonical_key points to existing object key.
        """
        fp = self.compressor.fingerprint(data)
        with self.lock:
            if fp in self.index:
                self.index[fp]["refs"] += 1
                self._persist_index()
                return fp, True, self.index[fp]["key"]
            # create new canonical_key if not provided
            canonical_key = suggested_key or f"obj_{fp}"
            self.index[fp] = {"key": canonical_key, "refs": 1}
            self._persist_index()
            return fp, False, canonical_key

    def ref(self, fingerprint: str):
        with self.lock:
            if fingerprint in self.index:
                self.index[fingerprint]["refs"] += 1
                self._persist_index()

    def unref(self, fingerprint: str) -> int:
        """
        Decrement ref count. Return remaining refs. If 0, caller should delete storage object.
        """
        with self.lock:
            if fingerprint not in self.index:
                return 0
            self.index[fingerprint]["refs"] -= 1
            refs = self.index[fingerprint]["refs"]
            if refs <= 0:
                # remove mapping — but don't delete underlying storage here
                del self.index[fingerprint]
            self._persist_index()
            return refs
