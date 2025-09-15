"""
Evaluation utilities:
 - Compute regression/classification metrics
 - Produce a standard evaluation report (persisted JSON)
 - Plot actual vs predicted (matplotlib)
"""

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .utils import logger, to_json, ensure_dir
import os


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def save_evaluation_report(metrics: Dict[str, Any], out_dir: str, name: str = "eval_report.json"):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, name)
    to_json(metrics, path)
    logger.info(f"Saved evaluation report to {path}")
    return path


def plot_actual_vs_pred(y_true, y_pred, out_path: str = None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    if out_path:
        ensure_dir(os.path.dirname(out_path) or ".")
        plt.savefig(out_path, bbox_inches="tight")
        logger.info(f"Saved Actual vs Predicted plot to {out_path}")
    plt.close()
