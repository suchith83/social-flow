"""
Prediction + scoring helpers for serving
- produce probabilities, expected uplift, and top-K retrieval
"""

import numpy as np
import pandas as pd
from .utils import load_pickle, logger

class ViralPredictor:
    def __init__(self, model=None, model_name="gbm_model.pkl"):
        if model is None:
            try:
                self.model = load_pickle(model_name)
            except Exception:
                self.model = None
                logger.warning("No model loaded; please pass model object at init or ensure model file exists.")
        else:
            self.model = model

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.predict_proba(X)

    def top_k_candidates(self, X_df: pd.DataFrame, k=100):
        """
        Return top-k items with highest viral probability.
        X_df: DataFrame with index=item_id aligned with features used for model.
        """
        probs = self.predict_proba(X_df.values)
        res = pd.DataFrame({
            "item_id": X_df.index,
            "viral_score": probs
        }).sort_values("viral_score", ascending=False).reset_index(drop=True)
        return res.head(k)
