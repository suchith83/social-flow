"""
Evaluation metrics for deep learning recommendation.
"""

import numpy as np


def precision_at_k(true_items, recommended_items, k=10):
    recommended_k = recommended_items[:k]
    return len(set(true_items) & set(recommended_k)) / k


def recall_at_k(true_items, recommended_items, k=10):
    recommended_k = recommended_items[:k]
    return len(set(true_items) & set(recommended_k)) / len(true_items)


def hit_rate(true_items, recommended_items, k=10):
    recommended_k = recommended_items[:k]
    return 1.0 if len(set(true_items) & set(recommended_k)) > 0 else 0.0
