"""
Unit tests for Scene Detection
"""

import unittest
from ai_models.content_analysis.scene_detection.detection import SceneDetector


class TestSceneDetection(unittest.TestCase):
    def test_threshold(self):
        detector = SceneDetector(threshold=0.5)
        self.assertAlmostEqual(detector.threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
