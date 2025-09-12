"""
age_policy.py

Defines age restriction policies and rules for different jurisdictions,
content types, and special cases. Centralized registry for age-related
compliance across the platform.
"""

from enum import Enum
from typing import Dict, Any


class ContentCategory(Enum):
    GENERAL = "general"
    TEEN = "teen"
    MATURE = "mature"
    ADULT = "adult"
    RESTRICTED = "restricted"


class Jurisdiction(Enum):
    GLOBAL = "global"
    US = "us"
    EU = "eu"
    INDIA = "india"
    UK = "uk"


class AgePolicy:
    """
    Central policy definition engine.
    Maps jurisdiction + category -> minimum required age.
    """

    _BASE_RULES: Dict[Jurisdiction, Dict[ContentCategory, int]] = {
        Jurisdiction.GLOBAL: {
            ContentCategory.GENERAL: 0,
            ContentCategory.TEEN: 13,
            ContentCategory.MATURE: 16,
            ContentCategory.ADULT: 18,
            ContentCategory.RESTRICTED: 21,
        },
        Jurisdiction.US: {
            ContentCategory.GENERAL: 0,
            ContentCategory.TEEN: 13,  # COPPA compliance
            ContentCategory.MATURE: 17,
            ContentCategory.ADULT: 18,
            ContentCategory.RESTRICTED: 21,
        },
        Jurisdiction.EU: {
            ContentCategory.GENERAL: 0,
            ContentCategory.TEEN: 16,  # GDPR stricter consent
            ContentCategory.MATURE: 18,
            ContentCategory.ADULT: 18,
            ContentCategory.RESTRICTED: 21,
        },
        Jurisdiction.INDIA: {
            ContentCategory.GENERAL: 0,
            ContentCategory.TEEN: 13,
            ContentCategory.MATURE: 18,
            ContentCategory.ADULT: 18,
            ContentCategory.RESTRICTED: 21,
        },
        Jurisdiction.UK: {
            ContentCategory.GENERAL: 0,
            ContentCategory.TEEN: 13,
            ContentCategory.MATURE: 16,
            ContentCategory.ADULT: 18,
            ContentCategory.RESTRICTED: 21,
        },
    }

    @classmethod
    def get_required_age(cls, category: ContentCategory, jurisdiction: Jurisdiction) -> int:
        """Get minimum required age for given content category and jurisdiction."""
        return cls._BASE_RULES.get(jurisdiction, cls._BASE_RULES[Jurisdiction.GLOBAL])[category]

    @classmethod
    def export_policies(cls) -> Dict[str, Any]:
        """Export all policies in serializable form."""
        return {
            jurisdiction.value: {
                category.value: age for category, age in categories.items()
            }
            for jurisdiction, categories in cls._BASE_RULES.items()
        }
