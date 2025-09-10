"""
Configuration for trending recommender system.
"""

import os
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Aggregation window
TREND_WINDOW = timedelta(days=1)   # last 24 hours by default
DECAY_FACTOR = 0.9                 # exponential decay for older interactions

# Recommendation
TOP_K = 10
