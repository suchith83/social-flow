"""
integration.py

API for other services to use moderation engine.
"""

from typing import Dict
from .moderator import Moderator
from .exceptions import ModerationViolation
from .enforcement import EnforcementEngine


class ModerationService:
    """Public API wrapper for moderation."""

    def __init__(self):
        self.moderator = Moderator()

    def check_and_moderate_text(self, user_id: str, content_id: str, text: str) -> Dict:
        try:
            return self.moderator.moderate_text(user_id, text)
        except ModerationViolation as v:
            EnforcementEngine.handle_violation(user_id, content_id, v)
            return {"status": "blocked", "reason": v.message}

    def check_and_moderate_image(self, user_id: str, content_id: str, image_bytes: bytes) -> Dict:
        try:
            return self.moderator.moderate_image(user_id, image_bytes)
        except ModerationViolation as v:
            EnforcementEngine.handle_violation(user_id, content_id, v)
            return {"status": "blocked", "reason": v.message}

    def check_and_moderate_video(self, user_id: str, content_id: str, video_path: str) -> Dict:
        try:
            return self.moderator.moderate_video(user_id, video_path)
        except ModerationViolation as v:
            EnforcementEngine.handle_violation(user_id, content_id, v)
            return {"status": "blocked", "reason": v.message}
