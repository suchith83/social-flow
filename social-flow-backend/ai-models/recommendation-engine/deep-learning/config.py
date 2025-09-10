"""
Configuration for deep learning recommendation engine.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = "cuda"  # "cuda" if GPU available else "cpu"

# Embeddings
EMBEDDING_DIM = 64

# Recommender
TOP_K = 10

# Evaluation
TEST_SPLIT = 0.2
RANDOM_SEED = 42
