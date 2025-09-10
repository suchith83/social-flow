"""
End-to-End Object Recognition Pipeline
"""

from .preprocessing import ImagePreprocessor
from .inference import InferenceEngine
from .evaluation import Evaluator
from .config import ObjectConfig
from .utils import setup_logger


class ObjectPipeline:
    def __init__(self, model_path: str, labels: list, device="cpu"):
        self.logger = setup_logger()
        self.preprocessor = ImagePreprocessor()
        self.inference = InferenceEngine(model_path, device)
        self.evaluator = Evaluator(labels)

    def run(self, filepaths: list, ground_truth: list = None):
        self.logger.info("Running object recognition pipeline...")
        preds = [self.inference.predict(fp) for fp in filepaths]

        if ground_truth:
            results = self.evaluator.evaluate(ground_truth, preds)
            self.logger.info(f"Evaluation: {results}")
            return results
        return preds
