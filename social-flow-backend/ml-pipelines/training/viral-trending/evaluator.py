# Metrics: AUC, F1, Precision@K, Viral-hit Recall
"""
evaluator.py
------------
Evaluation metrics for viral trending prediction.
"""

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np


def evaluate_classification(y_true, y_pred, threshold=0.5):
    preds_binary = (y_pred >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "f1": f1_score(y_true, preds_binary),
        "precision": precision_score(y_true, preds_binary),
        "recall": recall_score(y_true, preds_binary),
    }


def precision_at_k(y_true, y_pred, k=100):
    """Precision@K for viral-hit detection."""
    idx = np.argsort(-y_pred)[:k]
    return np.mean(y_true[idx])
