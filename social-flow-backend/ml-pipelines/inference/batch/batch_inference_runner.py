# Orchestrates batch inference jobs
# ================================================================
# File: batch_inference_runner.py
# Purpose: Entry point for orchestrating batch inference pipelines
# ================================================================

import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from data_loader import BatchDataLoader
from model_loader import ModelLoader
from inference_engine import InferenceEngine
from postprocessing import PostProcessor
from monitoring import Monitoring
from scheduler import Scheduler
from utils import load_config, setup_logger, retry

# Setup logging
logger = setup_logger("BatchInferenceRunner", level=logging.INFO)


class BatchInferenceRunner:
    """
    Orchestrates the end-to-end batch inference process.
    """

    def __init__(self, config_path: str = "configs/batch_inference.yaml"):
        self.config = load_config(config_path)
        self.monitoring = Monitoring(self.config["monitoring"])
        self.data_loader = BatchDataLoader(self.config["data"])
        self.model_loader = ModelLoader(self.config["model"])
        self.inference_engine = InferenceEngine(self.config["inference"])
