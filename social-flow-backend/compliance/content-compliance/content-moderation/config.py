"""
config.py

Configurable settings for content moderation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModerationConfig:
    ENABLE_STRICT_MODE: bool = True
    DEFAULT_LANGUAGE: str = "en"
    MAX_VIOLATIONS_BEFORE_BAN: int = 5
    AUTO_ESCALATE_SEVERITY: int = 3  # severity >= 3 auto-escalates
