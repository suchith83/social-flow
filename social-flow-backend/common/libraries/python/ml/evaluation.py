# common/libraries/python/ml/evaluation.py
"""
Evaluation metrics: classification, regression, ranking.
"""

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, ndcg_score
import numpy as np

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

def regression_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
    }

def ranking_metrics(y_true, y_score):
    return {
        "ndcg": ndcg_score(np.asarray([y_true]), np.asarray([y_score])),
    }
