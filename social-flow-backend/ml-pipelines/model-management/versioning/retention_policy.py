# GC and retention rules
# retention_policy.py
"""
Retention & garbage collection helpers.
Implements policies:
 - keep N latest versions per model
 - keep versions newer than T days
 - keep promoted tags (e.g., '*+staging', '*+production')
Performs safe delete: remove metadata entry and remove artifact from storage adapter.
"""

from typing import Dict, Any
from metadata_store import MetadataStore, SQLiteMetadataStore
from storage_adapter import StorageAdapter, LocalStorageAdapter
from utils import now_ts
import time
import logging
import re

logger = logging.getLogger("retention")


class RetentionPolicy:
    def __init__(self, store: MetadataStore, storage: StorageAdapter, keep_latest: int = 5, keep_days: Optional[int] = None, preserve_tags: Optional[list] = None):
        self.store = store
        self.storage = storage
        self.keep_latest = keep_latest
        self.keep_days = keep_days
        self.preserve_tags = preserve_tags or ["production", "staging"]

    def _is_promoted(self, version: str) -> bool:
        # promoted versions contain +tag per our promote strategy
        if "+" in version:
            tag = version.split("+", 1)[1]
            return tag in self.preserve_tags
        return False

    def collect(self, model_name: str):
        """
        Compute deletable versions and remove them.
        """
        versions = self.store.list_versions(model_name)
        deletable = []
        now = time.time()
        # skip promoted versions and keep_latest highest priority
        kept = []
        for idx, art in enumerate(versions):
            if self._is_promoted(art.version):
                kept.append(art)
            elif self.keep_days and (now - art.created_at) < (self.keep_days * 86400):
                kept.append(art)
            elif len(kept) < self.keep_latest:
                kept.append(art)
            else:
                deletable.append(art)

        # delete artifacts
        for art in deletable:
            try:
                logger.info(f"Deleting {art.name}:{art.version} -> {art.uri}")
                # try delete from storage: many adapters may not implement delete; attempt best-effort
                if hasattr(self.storage, "delete"):
                    self.storage.delete(art.uri)
                # remove metadata
                self.store.delete(art.name, art.version)
            except Exception as e:
                logger.error(f"Failed to delete {art.name}:{art.version}: {e}")
        return deletable
