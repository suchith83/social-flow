# Evaluate recommendation performance
# ============================
# File: evaluator.py
# ============================
import torch
import numpy as np

def compute_ranking_metrics(model, loader, device="cuda", k=10):
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            scores = torch.sigmoid(model(batch["user"], batch["item"])["logits"])
            y_scores.extend(scores.cpu().numpy())
            y_true.extend(batch["label"].cpu().numpy())

    y_true, y_scores = np.array(y_true), np.array(y_scores)
    # simple metrics
    auc = (y_true * y_scores).sum() / y_true.sum()
    precision_at_k = (y_true[np.argsort(-y_scores)[:k]].sum()) / k
    return {"auc_like": float(auc), f"precision@{k}": float(precision_at_k)}
