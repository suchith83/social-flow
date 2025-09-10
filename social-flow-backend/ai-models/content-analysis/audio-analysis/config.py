"""
Configuration module for audio-analysis

Handles environment variables, constants, and paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

class AudioConfig:
    """Configuration manager for audio analysis"""

    PROJECT_ROOT = Path(__file__).resolve().parent

    # Audio parameters
    SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
    N_MFCC = int(os.getenv("AUDIO_N_MFCC", 40))
    FRAME_LENGTH = float(os.getenv("AUDIO_FRAME_LENGTH", 0.025))  # 25ms
    FRAME_STEP = float(os.getenv("AUDIO_FRAME_STEP", 0.01))       # 10ms

    # Model parameters
    MODEL_TYPE = os.getenv("AUDIO_MODEL_TYPE", "cnn")
    NUM_CLASSES = int(os.getenv("AUDIO_NUM_CLASSES", 10))

    # Training parameters
    BATCH_SIZE = int(os.getenv("AUDIO_BATCH_SIZE", 32))
    LEARNING_RATE = float(os.getenv("AUDIO_LEARNING_RATE", 1e-4))
    EPOCHS = int(os.getenv("AUDIO_EPOCHS", 30))

    # Paths
    DATA_DIR = Path(os.getenv("AUDIO_DATA_DIR", PROJECT_ROOT / "data"))
    CHECKPOINT_DIR = Path(os.getenv("AUDIO_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
    LOG_DIR = Path(os.getenv("AUDIO_LOG_DIR", PROJECT_ROOT / "logs"))

    @classmethod
    def ensure_dirs(cls):
        """Ensure required directories exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Ensure dirs on import
AudioConfig.ensure_dirs()
