"""
config.py

Configurable options for copyright compliance workflows.
For production systems, these would be loaded from secure config / env / feature flags.
"""

from dataclasses import dataclass
from pathlib import Path

LOG_BASE = Path("logs/compliance/copyright")
EVIDENCE_BASE = Path("storage/evidence/copyright")

@dataclass(frozen=True)
class CopyrightConfig:
    """Global toggles and thresholds."""
    ENABLE_AUTO_TAKEDOWN: bool = True          # whether to auto-remove when rule says so
    AUTO_TAKEDOWN_DELAY_SECONDS: int = 0       # delay before takedown (0 = immediate)
    MAX_PENDING_NOTICES_PER_USER: int = 10
    EVIDENCE_STORE_PATH: str = str(EVIDENCE_BASE)
    ESCALATION_SEVERITY_THRESHOLD: int = 75    # score above which auto-escalate to legal
