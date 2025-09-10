"""
End-to-End Scene Detection Pipeline
"""

from .preprocessing import VideoPreprocessor
from .detection import SceneDetector
from .inference import InferenceEngine
from .evaluation import Evaluator
from .utils import setup_logger


class ScenePipeline:
    def __init__(self, model_path: str, labels: list, device="cpu"):
        self.logger = setup_logger()
        self.preprocessor = VideoPreprocessor()
        self.detector = SceneDetector()
        self.inference = InferenceEngine(model_path, device)
        self.evaluator = Evaluator(labels)

    def run(self, video_path: str, ground_truth=None):
        self.logger.info("Running scene detection pipeline...")
        scene_boundaries = self.detector.detect_scenes(video_path)
        preds = self.inference.classify_frames(video_path)

        if ground_truth:
            results = self.evaluator.evaluate(ground_truth, preds)
            self.logger.info(f"Evaluation: {results}")
            return {"boundaries": scene_boundaries, "evaluation": results}
        return {"boundaries": scene_boundaries, "predictions": preds}
