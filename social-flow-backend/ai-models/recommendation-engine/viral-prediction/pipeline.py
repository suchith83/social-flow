"""
End-to-end pipeline:
- build dataset
- features
- train model
- evaluate
- produce top-K predictions
"""

import numpy as np
import pandas as pd
from .data_prep import build_dataset
from .feature_engineering import assemble_feature_matrix
from .trainer import ViralTrainer
from .predictor import ViralPredictor
from .evaluation import classification_metrics, lift_at_k
from .utils import logger, set_seed

class ViralPredictionPipeline:
    def __init__(self, model_type="gbm", seed=None):
        set_seed(seed)
        self.model_type = model_type
        self.trainer = ViralTrainer(model_type=model_type, seed=seed)
        self.predictor = None
        self.scaler = None

    def prepare_and_train(self, interactions: pd.DataFrame, meta: pd.DataFrame = None, content_col="text"):
        # 1. build item-level dataset and labels
        item_df = build_dataset(interactions, meta)

        # 2. feature engineering
        X_df, y, scaler = assemble_feature_matrix(item_df, meta_df=meta, content_col=content_col)
        self.scaler = scaler

        X = X_df.values
        y = y.values.astype(int)

        # 3. train model
        model, auc = self.trainer.train(X, y)
        # reload model wrapper if saved
        self.predictor = ViralPredictor(model=model)
        return {"auc": auc, "trained_model": model, "X_df": X_df, "y": y}

    def evaluate(self, X_df: pd.DataFrame, y: np.ndarray, top_k=100):
        if self.predictor is None:
            raise RuntimeError("No predictor; train first")
        probs = self.predictor.predict_proba(X_df.values)
        metrics = classification_metrics(y, probs)
        lift = lift_at_k(y, probs, k=top_k)
        metrics["lift_at_k"] = lift
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def recommend_top_k(self, X_df: pd.DataFrame, k=100):
        if self.predictor is None:
            raise RuntimeError("No predictor; train first")
        return self.predictor.top_k_candidates(X_df, k=k)
