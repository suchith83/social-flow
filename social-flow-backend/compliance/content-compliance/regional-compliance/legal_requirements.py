"""
legal_requirements.py

Mapping of legal obligations per jurisdiction:
- data retention minima
- notice windows for takedown responses
- mandatory reporting (e.g., child exploitation reporting)
This mapping is intentionally conservative and example-only.
"""

from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LegalRequirement:
    description: str
    params: Dict[str, Any]

# Example table
_LEGAL_TABLE = {
    "eu": {
        "retention_min_days": LegalRequirement("Min retention for transactional logs", {"days": 180}),
        "notice_response_days": LegalRequirement("Time to respond to formal takedown", {"days": 14})
    },
    "us": {
        "retention_min_days": LegalRequirement("Min retention for transactional logs", {"days": 90}),
        "notice_response_days": LegalRequirement("Time to respond to DMCA notice", {"days": 14})
    },
    "india": {
        "retention_min_days": LegalRequirement("Min retention for lawful access", {"days": 180}),
        "notice_response_days": LegalRequirement("Time to respond to takedown", {"days": 7})
    }
}

def get_legal_requirements_for_jurisdiction(jurisdiction: str) -> Dict[str, LegalRequirement]:
    return _LEGAL_TABLE.get(jurisdiction.lower(), {})
