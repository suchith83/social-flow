"""
Configuration for viral prediction pipeline
"""

import os
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Problem setup
EARLY_WINDOW = timedelta(hours=6)      # use first 6 hours of interactions as input
PREDICT_HORIZON = timedelta(days=7)    # predict growth over next 7 days
VIRAL_THRESHOLD = 5.0                  # e.g., growth factor or relative percent to label 'viral'

# Training
RANDOM_SEED = 42
TEST_SPLIT = 0.2
CV_FOLDS = 5

# Modeling
GBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

NN_PARAMS = {
    "hidden_dims": [256, 128],
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 256,
    "epochs": 20,
}

# Misc
TOP_K = 100
