# Evaluation metrics (NDCG, MAP, Recall@K, etc.)
"""
evaluator.py
------------
Provides RecSys evaluation metrics: NDCG, Recall@K, MAP.
"""

import numpy as np


def recall_at_k(y_true, y_pred, k=10):
    """Recall@K"""
    y_pred_topk = y_pred[:k]
    return len(set(y_true) & set(y_pred_topk)) / len(set(y_true))


def ndcg_at_k(y_true, y_pred, k=10):
    """Normalized Discounted Cumulative Gain at K"""
    y_pred_topk = y_pred[:k]
    dcg = sum([1 / np.log2(i + 2) for i, p in enumerate(y_pred_topk) if p in y_true])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(y_true), k))])
    return dcg / idcg if idcg > 0 else 0


def map_at_k(y_true, y_pred, k=10):
    """Mean Average Precision at K"""
    score = 0.0
    hits = 0
    for i, p in enumerate(y_pred[:k]):
        if p in y_true:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(y_true), k)


if __name__ == "__main__":
    y_true = [1, 2, 3]
    y_pred = [3, 4, 1, 2, 5]
    print("Recall@3:", recall_at_k(y_true, y_pred, 3))
    print("NDCG@3:", ndcg_at_k(y_true, y_pred, 3))
    print("MAP@3:", map_at_k(y_true, y_pred, 3))
