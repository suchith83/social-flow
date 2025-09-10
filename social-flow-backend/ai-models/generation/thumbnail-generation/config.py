"""
Config constants for thumbnail-generation package.
Change these to tune training and inference behaviour.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Image sizes
SRC_SIZE = (1024, 576)       # source frame resolution (W,H) typical wide format
THUMBNAIL_SIZES = [(320, 180), (480, 270), (1280, 720)]  # supported output sizes (w,h)

# Training hyperparams
BATCH_SIZE = 16
LR = 2e-4
EPOCHS = 40
DEVICE = "cuda"  # auto-fallback to cpu in code

# Model architecture
ENCODER_FEATURES = [64, 128, 256, 512]
DECODER_FEATURES = [512, 256, 128, 64]
LATENT_DIM = 512

# GAN hyperparams (if using adversarial)
USE_GAN = True
GAN_LOSS_WEIGHT = 0.01
PERCEPTUAL_LOSS_WEIGHT = 1.0
PIXEL_LOSS_WEIGHT = 1.0

# Misc
SEED = 42
LOG_INTERVAL = 50
SAVE_FREQ = 5  # epochs
