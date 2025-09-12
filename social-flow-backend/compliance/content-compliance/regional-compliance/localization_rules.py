"""
localization_rules.py

Locale- and language-based moderation adjustments.
Examples:
- Different profanity lists by locale
- Language-specific allowed expressions
- Content label translations for moderation notices
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Minimal example sets; in production these are large curated lists or ML models.
_LOCALE_PROFANITY_OVERRIDES: Dict[str, List[str]] = {
    "fr": ["merde", "putain"],
    "en-uk": ["bloody"],
}

# Content label translations (for user-facing messages)
_LABEL_TRANSLATIONS = {
    "en": {
        "blocked_reason_age": "You must be older to view this content.",
        "blocked_reason_residency": "Content is not available in your region."
    },
    "fr": {
        "blocked_reason_age": "Vous devez être plus âgé pour voir ce contenu.",
        "blocked_reason_residency": "Le contenu n'est pas disponible dans votre région."
    }
}

def get_locale_profanity(locale: str) -> List[str]:
    return _LOCALE_PROFANITY_OVERRIDES.get(locale.lower(), [])

def translate_label(key: str, locale: str = "en") -> str:
    return _LABEL_TRANSLATIONS.get(locale.split("-")[0], _LABEL_TRANSLATIONS["en"]).get(key, key)
