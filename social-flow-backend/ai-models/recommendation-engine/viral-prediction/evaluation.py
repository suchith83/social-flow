"""
Evaluation functions: classification metrics, calibration, lift curves, business KPIs
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score

def classification_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"auc": auc, "average_precision": ap, "precision": p, "recall": r, "f1": f1}

def lift_at_k(y_true, scores, k=100):
    """
    Compute lift at top-K relative to random baseline.
    """
    idx_sorted = np.argsort(scores)[::-1][:k]
    topk_rate = np.mean(y_true[idx_sorted])
    overall_rate = np.mean(y_true)
    if overall_rate == 0:
        return np.inf
    return topk_rate / overall_rate

def calibration_bucket_stats(y_true, probs, n_buckets=10):
    bins = np.linspace(0.0, 1.0, n_buckets+1)
    inds = np.digitize(probs, bins) - 1
    stats = []
    for b in range(n_buckets):
        mask = inds == b
        if mask.sum() == 0:
            stats.append({"bucket": b, "count": 0, "mean_pred": np.nan, "mean_true": np.nan})
        else:
            stats.append({
                "bucket": b,
                "count": int(mask.sum()),
                "mean_pred": float(np.mean(probs[mask])),
                "mean_true": float(np.mean(y_true[mask]))
            })
    return stats
