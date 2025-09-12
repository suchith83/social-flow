"""
ml_detector.py

ML-based classifiers for detecting moderation categories.
Stubbed with simulated logic; replace with production AI models.
"""

import random
from .content_policy import ModerationCategory


class MLDetector:
    """Fake ML-based classifier (replace with real models)."""

    def classify_text(self, text: str) -> ModerationCategory:
        text_lower = text.lower()
        if "hate" in text_lower:
            return ModerationCategory.HATE_SPEECH
        elif "kill" in text_lower:
            return ModerationCategory.VIOLENCE
        elif "xxx" in text_lower:
            return ModerationCategory.ADULT
        elif "spam" in text_lower:
            return ModerationCategory.SPAM
        return ModerationCategory.SAFE

    def classify_image(self, image_bytes: bytes) -> ModerationCategory:
        # Stubbed random classification
        return random.choice(list(ModerationCategory))

    def classify_video(self, video_path: str) -> ModerationCategory:
        # Stubbed random classification
        return random.choice(list(ModerationCategory))
