"""
takedown_manager.py

Implements takedown lifecycle:
- accept notice
- store evidence
- fingerprint content
- decide action based on policy
- execute takedown (soft delete / remove visibility)
- notify parties and log evidence
"""

from typing import Optional, Dict, Any
import time
from datetime import datetime

from .notices import validate_notice_payload, Notice
from .evidence_store import store_evidence, retrieve_evidence
from .content_fingerprinter import sha256_file, average_hash_image_bytes, video_frame_hash_stub
from .copyright_policy import CopyrightPolicy, CopyrightCategory
from .config import CopyrightConfig
from .audit_logger import AuditLogger
from .exceptions import TakedownAlreadyProcessedError, InvalidNoticeError


# In-memory state store for demo purposes. Replace with persistent DB in production.
_TAKEDOWN_DB: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


class TakedownManager:
    """Manager for takedown requests and transitions."""

    @staticmethod
    def submit_notice(payload: Dict[str, Any], attachment_bytes: Optional[bytes] = None, attachment_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit a takedown notice, store evidence, and perform initial triage.
        Returns an envelope with notice_id and triage result.
        """
        # Validate input
        notice = validate_notice_payload(payload)

        # Persist evidence to filesystem
        evidence_meta = {"complainant": notice.complainant_id, "original_notice": notice.to_dict()}
        notice_dir = store_evidence(notice.id, attachment_bytes, attachment_filename, evidence_meta)

        # Basic fingerprinting depending on file type
        fingerprint = None
        if attachment_bytes and attachment_filename:
            ext = attachment_filename.split(".")[-1].lower()
            if ext in ("jpg", "jpeg", "png", "gif", "webp"):
                try:
                    fingerprint = average_hash_image_bytes(attachment_bytes)
                except Exception:
                    fingerprint = None
            elif ext in ("mp4", "mov", "mkv", "webm"):
                # in real system, write bytes to temp file and compute frame hashes
                fingerprint = video_frame_hash_stub(str(notice_dir / attachment_filename))
            else:
                # fallback to sha256 hex of bytes
                fingerprint = sha256_file(str(notice_dir / attachment_filename)) if (notice_dir / attachment_filename).exists() else None

        # Persist takedown entry
        entry = {
            "notice": notice.to_dict(),
            "evidence_path": str(notice_dir),
            "fingerprint": fingerprint,
            "status": "submitted",
            "triage": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        _TAKEDOWN_DB[notice.id] = entry

        AuditLogger.log_event("NOTICE_SUBMITTED", notice.complainant_id, {"notice_id": notice.id, "infringing_content_id": notice.infringing_content_id})

        # Perform automatic triage (policy + score)
        triage = TakedownManager._triage(entry)
        entry["triage"] = triage
        entry["updated_at"] = _now_iso()

        # Apply action if policy decides so
        if triage["action"] == "remove" and triage["auto_action"]:
            TakedownManager._execute_takedown(entry["notice"]["id"], entry["notice"]["infringing_content_id"], triage["reason"])

        return {"notice_id": notice.id, "triage": triage, "status": entry["status"]}

    @staticmethod
    def _triage(entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine probable category & action.
        This is a simplified heuristic: use metadata + fingerprint presence to determine likely category.
        """
        # Very simplistic heuristics for demo:
        notice = entry["notice"]
        desc = (notice.get("description") or "").lower()
        fingerprint = entry.get("fingerprint")
        probable_category = CopyrightCategory.UNKNOWN

        if "unauthorized" in desc or "copyright" in desc or "stolen" in desc or fingerprint:
            probable_category = CopyrightCategory.DIRECT_INFRINGEMENT
        elif "license" in desc or "cc-by" in desc or "creative commons" in desc:
            probable_category = CopyrightCategory.LICENSE_VIOLATION
        elif "parody" in desc or "transformative" in desc:
            probable_category = CopyrightCategory.POTENTIAL_FAIR_USE
        else:
            probable_category = CopyrightCategory.UNKNOWN

        rule = CopyrightPolicy.get_rule(probable_category)
        severity = rule.severity_score
        auto_action = rule.remove_immediately and CopyrightConfig.ENABLE_AUTO_TAKEDOWN

        # Decide action string: remove | review | monitor
        action = "remove" if rule.remove_immediately else "review"

        triage = {
            "category": probable_category.value,
            "severity": severity,
            "reason": f"heuristic: matched keywords or fingerprint present" if fingerprint or "copyright" in desc else "heuristic: no strong signals",
            "action": action,
            "auto_action": auto_action,
            "fingerprint": fingerprint,
        }

        AuditLogger.log_event("TRIAGE_COMPLETE", notice["complainant_id"], {"notice_id": notice["id"], "triage": triage})
        return triage

    @staticmethod
    def _execute_takedown(notice_id: str, content_id: str, reason: str) -> None:
        """
        Execute takedown action against content.
        In a real platform this would call content services to mark content as removed/hidden.
        Here we just record the action and log it.
        """
        entry = _TAKEDOWN_DB.get(notice_id)
        if not entry:
            raise InvalidNoticeError(f"Notice {notice_id} not found")

        if entry.get("status") in ("removed", "closed"):
            raise TakedownAlreadyProcessedError(f"Notice {notice_id} already processed with status {entry.get('status')}")

        # Simulate delay (respect AUTO_TAKEDOWN_DELAY_SECONDS)
        delay = CopyrightConfig.AUTO_TAKEDOWN_DELAY_SECONDS
        if delay > 0:
            time.sleep(delay)

        # Mark as removed
        entry["status"] = "removed"
        entry["removed_at"] = _now_iso()
        entry["remove_reason"] = reason
        entry["updated_at"] = _now_iso()

        # Log audit and notify owners (stubs)
        AuditLogger.log_event("CONTENT_REMOVED", "system", {"notice_id": notice_id, "content_id": content_id, "reason": reason})
        # In prod: call notifications / webhooks here

    @staticmethod
    def get_notice(notice_id: str) -> Dict[str, Any]:
        entry = _TAKEDOWN_DB.get(notice_id)
        if not entry:
            raise InvalidNoticeError(f"Notice {notice_id} not found")
        return entry

    @staticmethod
    def restore_content(notice_id: str, admin_id: str, reason: str) -> None:
        """
        Restore content previously removed. Only admin should call this.
        """
        entry = _TAKEDOWN_DB.get(notice_id)
        if not entry:
            raise InvalidNoticeError(f"Notice {notice_id} not found")

        if entry.get("status") != "removed":
            raise TakedownAlreadyProcessedError(f"Notice {notice_id} is not in removed state.")

        entry["status"] = "restored"
        entry["restored_by"] = admin_id
        entry["restored_reason"] = reason
        entry["restored_at"] = _now_iso()
        entry["updated_at"] = _now_iso()

        AuditLogger.log_event("CONTENT_RESTORED", admin_id, {"notice_id": notice_id, "reason": reason})
