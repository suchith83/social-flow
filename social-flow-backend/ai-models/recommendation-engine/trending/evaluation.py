"""
Evaluation metrics for trending recommender.
"""

import numpy as np


def coverage(recommended_items, all_items):
    return len(set(recommended_items)) / len(set(all_items))


def novelty(recommended_items, popular_items):
    """
    Novelty = proportion of items not in most popular.
    """
    return len(set(recommended_items) - set(popular_items)) / len(recommended_items)


def hit_rate(recommended_items, test_items):
    return 1.0 if len(set(recommended_items) & set(test_items)) > 0 else 0.0
