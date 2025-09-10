"""
Configuration for Object Recognition
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class ObjectConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent

    # Image settings
    IMG_SIZE = int(os.getenv("OBJ_IMG_SIZE", 224))
    CHANNELS = 3

    # Model settings
    MODEL_TYPE = os.getenv("OBJ_MODEL_TYPE", "resnet50")  # options: resnet50, vit
    NUM_CLASSES = int(os.getenv("OBJ_NUM_CLASSES", 80))   # COCO default

    # Training
    BATCH_SIZE = int(os.getenv("OBJ_BATCH_SIZE", 32))
    LEARNING_RATE = float(os.getenv("OBJ_LEARNING_RATE", 1e-4))
    EPOCHS = int(os.getenv("OBJ_EPOCHS", 20))

    # Paths
    DATA_DIR = Path(os.getenv("OBJ_DATA_DIR", PROJECT_ROOT / "data"))
    CHECKPOINT_DIR = Path(os.getenv("OBJ_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
    LOG_DIR = Path(os.getenv("OBJ_LOG_DIR", PROJECT_ROOT / "logs"))

    @classmethod
    def ensure_dirs(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

ObjectConfig.ensure_dirs()
