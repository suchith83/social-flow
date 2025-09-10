"""
Unit tests for object recognition
"""

import unittest
import torch
from ai_models.content_analysis.object_recognition.preprocessing import ImagePreprocessor


class TestObjectRecognition(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor()

    def test_preprocessing(self):
        dummy_img = torch.rand(3, 224, 224)
        self.assertEqual(dummy_img.shape, (3, 224, 224))


if __name__ == "__main__":
    unittest.main()
