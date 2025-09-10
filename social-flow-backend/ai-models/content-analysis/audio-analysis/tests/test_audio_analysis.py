"""
Unit tests for audio-analysis
"""

import unittest
import numpy as np
from ai_models.content_analysis.audio_analysis.preprocessing import AudioPreprocessor
from ai_models.content_analysis.audio_analysis.feature_extraction import FeatureExtractor


class TestAudioAnalysis(unittest.TestCase):

    def setUp(self):
        self.preprocessor = AudioPreprocessor()
        self.extractor = FeatureExtractor()

    def test_preprocessing_pipeline(self):
        y = np.random.randn(16000)  # 1 sec fake audio
        y_norm = self.preprocessor.normalize(y)
        self.assertAlmostEqual(np.max(np.abs(y_norm)), 1.0, delta=1e-6)

    def test_feature_extraction(self):
        y = np.random.randn(16000)
        mfcc = self.extractor.extract_mfcc(y)
        self.assertEqual(len(mfcc.shape), 2)


if __name__ == "__main__":
    unittest.main()
