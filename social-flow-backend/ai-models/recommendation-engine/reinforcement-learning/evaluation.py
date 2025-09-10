"""
Evaluation metrics for RL recommender.
"""

import numpy as np


def average_reward(rewards):
    return np.mean(rewards)


def success_rate(rewards, threshold=1):
    return np.mean([1 if r >= threshold else 0 for r in rewards])
