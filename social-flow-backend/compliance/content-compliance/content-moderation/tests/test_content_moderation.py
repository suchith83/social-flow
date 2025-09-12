"""
test_content_moderation.py

Tests for content moderation module.
"""

import unittest
from compliance.content_compliance.content_moderation.integration import ModerationService


class TestContentModeration(unittest.TestCase):
    def setUp(self):
        self.service = ModerationService()

    def test_safe_text(self):
        result = self.service.check_and_moderate_text("u1", "c1", "Hello world!")
        self.assertIn("category", result)

    def test_hate_text_violation(self):
        result = self.service.check_and_moderate_text("u1", "c2", "I hate everyone")
        self.assertEqual(result["status"], "blocked")

    def test_video_moderation(self):
        result = self.service.check_and_moderate_video("u1", "c3", "fake_video.mp4")
        self.assertIn("category", result)


if __name__ == "__main__":
    unittest.main()
