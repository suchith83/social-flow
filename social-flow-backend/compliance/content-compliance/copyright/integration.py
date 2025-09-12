"""
integration.py

Public API for other platform services to interact with the copyright subsystem.
Provides:
- submit_notice(...)
- get_notice(...)
- restore_content(...)
- basic verify endpoints for fingerprints
"""

from typing import Optional, Dict, Any
from datetime import date

from .takedown_manager import TakedownManager
from .evidence_store import retrieve_evidence
from .content_fingerprinter import sha256_file


class CopyrightService:
    """Facade to be used by higher-level services (API, background jobs)."""

    def __init__(self):
        self.manager = TakedownManager()

    def submit_notice(self, payload: Dict[str, Any], attachment_bytes: Optional[bytes] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """Submit a takedown/notice with optional evidence attachment bytes."""
        return self.manager.submit_notice(payload, attachment_bytes, filename)

    def get_notice(self, notice_id: str) -> Dict[str, Any]:
        """Get stored notice + triage + evidence summary."""
        entry = self.manager.get_notice(notice_id)
        # attach evidence summary
        evidence_info = retrieve_evidence(notice_id)
        entry = entry.copy()
        entry["evidence_summary"] = evidence_info
        return entry

    def restore_content(self, notice_id: str, admin_id: str, reason: str) -> None:
        """Admin restore a previously removed content item."""
        return self.manager.restore_content(notice_id, admin_id, reason)

    def verify_file_fingerprint(self, path: str) -> str:
        """Utility to compute sha256 of a file for cross-checking evidence."""
        return sha256_file(path)
