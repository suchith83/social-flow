"""
Audio Analysis Pipeline

Orchestrates:
- Preprocessing
- Feature extraction
- Inference
- Evaluation
"""

from .preprocessing import AudioPreprocessor
from .feature_extraction import FeatureExtractor
from .inference import InferenceEngine
from .evaluation import Evaluator
from .config import AudioConfig
from .utils import setup_logger


class AudioPipeline:
    def __init__(self, model_path: str, labels: list, device="cpu"):
        self.logger = setup_logger()
        self.preprocessor = AudioPreprocessor()
        self.extractor = FeatureExtractor()
        self.inference = InferenceEngine(model_path, device=device)
        self.evaluator = Evaluator(labels)

    def run(self, filepaths: list, ground_truth: list = None):
        self.logger.info("Starting audio pipeline...")
        preds = [self.inference.predict(fp) for fp in filepaths]

        if ground_truth:
            results = self.evaluator.evaluate(ground_truth, preds)
            self.logger.info(f"Evaluation: {results}")
            return results
        return preds
