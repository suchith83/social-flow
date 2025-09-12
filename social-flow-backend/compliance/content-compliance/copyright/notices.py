"""
notices.py

Notice models and validation for takedown complaints (DMCA-like).
Includes parsers, canonicalization and basic rate-limiting checks.
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from .exceptions import InvalidNoticeError


@dataclass
class Notice:
    """
    Canonical notice model.
    - complainant_id: id of the submitting user/agent
    - claimed_owner: description of the claimed copyright owner
    - infringing_content_id: id of the content on platform
    - description: free-text description of infringement
    - attachment_filename: optional evidentiary file name stored separately
    - created_at: timestamp
    - id: unique notice id assigned by system
    - jurisdiction: optional string for jurisdiction-specific processing
    """
    complainant_id: str
    claimed_owner: str
    infringing_content_id: str
    description: str
    attachment_filename: Optional[str] = None
    jurisdiction: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = d["created_at"].isoformat()
        return d


def validate_notice_payload(payload: Dict[str, Any]) -> Notice:
    """Validate incoming payload shape and required fields. Raise InvalidNoticeError on failure."""
    required = ["complainant_id", "claimed_owner", "infringing_content_id", "description"]
    missing = [r for r in required if r not in payload or not payload[r]]
    if missing:
        raise InvalidNoticeError(f"Missing required fields in notice: {', '.join(missing)}")

    # Length and content checks
    if len(payload["description"]) < 10:
        raise InvalidNoticeError("Description is too short to constitute a valid notice.")

    # Can add identity verification check hooks here (e.g., check complainant email verified)
    return Notice(
        complainant_id=payload["complainant_id"],
        claimed_owner=payload["claimed_owner"],
        infringing_content_id=payload["infringing_content_id"],
        description=payload["description"],
        attachment_filename=payload.get("attachment_filename"),
        jurisdiction=payload.get("jurisdiction"),
    )
