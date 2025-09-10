"""
Configuration for RL-based recommender system.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# RL environment
MAX_STEPS = 50
EPISODES = 500

# Agent hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE_FREQ = 50

# Device
DEVICE = "cuda"
