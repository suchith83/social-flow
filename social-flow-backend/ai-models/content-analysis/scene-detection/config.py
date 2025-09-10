"""
Configuration for Scene Detection
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class SceneConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent

    # Video processing
    FRAME_RATE = int(os.getenv("SCENE_FRAME_RATE", 2))  # sample 2 FPS
    IMG_SIZE = int(os.getenv("SCENE_IMG_SIZE", 224))

    # Model settings
    MODEL_TYPE = os.getenv("SCENE_MODEL_TYPE", "resnet18")
    NUM_CLASSES = int(os.getenv("SCENE_NUM_CLASSES", 10))

    # Thresholds
    HIST_DIFF_THRESHOLD = float(os.getenv("SCENE_HIST_DIFF_THRESHOLD", 0.5))

    # Paths
    DATA_DIR = Path(os.getenv("SCENE_DATA_DIR", PROJECT_ROOT / "data"))
    CHECKPOINT_DIR = Path(os.getenv("SCENE_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
    LOG_DIR = Path(os.getenv("SCENE_LOG_DIR", PROJECT_ROOT / "logs"))

    @classmethod
    def ensure_dirs(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

SceneConfig.ensure_dirs()
