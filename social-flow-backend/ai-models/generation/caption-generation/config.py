"""
Configuration for Caption Generation Engine
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 30

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = "cuda"  # set "cpu" if no GPU

# Model
EMBED_DIM = 512
HIDDEN_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1

# Decoding
BEAM_SIZE = 5
MAX_CAPTION_LEN = 30
