"""
Configuration file for Content-Based Recommendation Engine.
Stores constants, thresholds, and model paths.
"""

import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Feature extraction
MAX_FEATURES = 5000       # Max vocab size for text vectorizers
EMBEDDING_DIM = 300       # Dimensionality for word embeddings

# Similarity
SIMILARITY_METRIC = "cosine"  # Options: cosine, dot, euclidean

# Recommendations
TOP_K = 10                # Number of items to recommend

# Evaluation
TEST_SPLIT = 0.2
RANDOM_SEED = 42
