"""
moderator.py

Core moderation engine that applies policies and ML detectors
to classify and validate user content.
"""

from typing import Dict
from .content_policy import ModerationCategory, ContentPolicy, SeverityLevel
from .exceptions import ModerationViolation
from .ml_detector import MLDetector


class Moderator:
    """Moderates text and content against rules."""

    def __init__(self):
        self.detector = MLDetector()

    def moderate_text(self, user_id: str, text: str) -> Dict:
        """
        Classify text into moderation category and enforce severity.
        """
        category = self.detector.classify_text(text)
        severity = ContentPolicy.get_severity(category)

        if severity.value >= SeverityLevel.HIGH.value:
            raise ModerationViolation(
                f"Content violates {category.value.upper()} rules with severity {severity.name}."
            )

        return {"category": category.value, "severity": severity.name}

    def moderate_image(self, user_id: str, image_bytes: bytes) -> Dict:
        """Moderate images using ML classifier."""
        category = self.detector.classify_image(image_bytes)
        severity = ContentPolicy.get_severity(category)

        if severity == SeverityLevel.CRITICAL:
            raise ModerationViolation(
                f"Image violates {category.value.upper()} rules with severity {severity.name}."
            )

        return {"category": category.value, "severity": severity.name}

    def moderate_video(self, user_id: str, video_path: str) -> Dict:
        """Moderate video content."""
        category = self.detector.classify_video(video_path)
        severity = ContentPolicy.get_severity(category)

        if severity.value >= SeverityLevel.HIGH.value:
            raise ModerationViolation(
                f"Video violates {category.value.upper()} rules with severity {severity.name}."
            )

        return {"category": category.value, "severity": severity.name}
