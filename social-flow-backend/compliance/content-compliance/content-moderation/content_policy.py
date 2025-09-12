"""
content_policy.py

Defines moderation rules, categories, and severity levels
for platform compliance. Central registry for content rules.
"""

from enum import Enum
from typing import Dict


class ModerationCategory(Enum):
    SAFE = "safe"
    SPAM = "spam"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    ADULT = "adult"
    COPYRIGHT = "copyright"
    OTHER = "other"


class SeverityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ContentPolicy:
    """Defines severity levels per category."""

    _RULES: Dict[ModerationCategory, SeverityLevel] = {
        ModerationCategory.SAFE: SeverityLevel.LOW,
        ModerationCategory.SPAM: SeverityLevel.MEDIUM,
        ModerationCategory.HATE_SPEECH: SeverityLevel.CRITICAL,
        ModerationCategory.VIOLENCE: SeverityLevel.CRITICAL,
        ModerationCategory.ADULT: SeverityLevel.HIGH,
        ModerationCategory.COPYRIGHT: SeverityLevel.HIGH,
        ModerationCategory.OTHER: SeverityLevel.MEDIUM,
    }

    @classmethod
    def get_severity(cls, category: ModerationCategory) -> SeverityLevel:
        return cls._RULES.get(category, SeverityLevel.MEDIUM)

    @classmethod
    def export_policies(cls) -> Dict[str, int]:
        return {cat.value: sev.value for cat, sev in cls._RULES.items()}
