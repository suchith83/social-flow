"""
Evaluation metrics for content-based recommender.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def precision_at_k(true_items, recommended_items, k=10):
    recommended_k = recommended_items[:k]
    return len(set(true_items) & set(recommended_k)) / k


def recall_at_k(true_items, recommended_items, k=10):
    recommended_k = recommended_items[:k]
    return len(set(true_items) & set(recommended_k)) / len(true_items)


def f1_at_k(true_items, recommended_items, k=10):
    p = precision_at_k(true_items, recommended_items, k)
    r = recall_at_k(true_items, recommended_items, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)
