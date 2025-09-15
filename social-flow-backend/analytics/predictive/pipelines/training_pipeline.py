"""
Training Pipeline
- Loads processed features, trains a model using analytics.predictive.models.Trainer
- Registers model with model_registry and optionally logs to MLflow
- Exposes run() and dry_run()
"""

from typing import Optional
import os
import pandas as pd
from .config import settings
from .utils import logger, ensure_dir, write_json
from analytics.predictive.models.data_loader import DataLoader
from analytics.predictive.models.model_trainer import Trainer
from analytics.predictive.models.model_registry import register_model
import joblib

class TrainingPipeline:
    name = "training"

    def __init__(
        self,
        feature_table: str = "raw_events",
        model_name: Optional[str] = None,
        train_suffix: str = "_train.parquet",
        val_suffix: str = "_val.parquet",
        numeric_features: list | None = None,
        categorical_features: list | None = None,
        param_distributions: dict | None = None,
        n_iter: int = 20,
    ):
        self.feature_table = feature_table
        self.model_name = model_name or "user_growth_xgb"
        self.train_path = os.path.join(settings.FEATURE_STORE, f"{feature_table}{train_suffix}")
        self.val_path = os.path.join(settings.FEATURE_STORE, f"{feature_table}{val_suffix}")
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.trainer = Trainer(model_dir=settings.MODEL_DIR)

    def dry_run(self):
        logger.info("Training dry-run checking training artifacts presence")
        if not os.path.exists(self.train_path) or not os.path.exists(self.val_path):
            logger.warning(f"Train/val files missing. train:{self.train_path}, val:{self.val_path}")

    def run(self):
        # Load train/val
        if not os.path.exists(self.train_path) or not os.path.exists(self.val_path):
            raise FileNotFoundError("Processed feature files not found. Run preprocessing first.")

        train_df = pd.read_parquet(self.train_path)
        val_df = pd.read_parquet(self.val_path)

        # separate X/y
        # assume last column is target (persisted by PreprocessingPipeline)
        y_col = train_df.columns[-1]
        X_train = train_df.drop(columns=[y_col])
        y_train = train_df[y_col]
        X_val = val_df.drop(columns=[y_col])
        y_val = val_df[y_col]

        # Kick off training (Trainer will persist model in settings.MODEL_DIR)
        pipeline, metrics = self.trainer.train_xgb_regressor(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            numeric_features=self.numeric_features or [],
            categorical_features=self.categorical_features or [],
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
        )

        # Register model in registry
        artifact_name = os.path.join(settings.MODEL_DIR, f"{self.model_name}.pkl")
        # Trainer already saved model with timestamp; find newest matching file
        # Simple approach: save the last pipeline to a stable name also
        stable_path = os.path.join(settings.MODEL_DIR, f"{self.model_name}_latest.pkl")
        joblib.dump(pipeline, stable_path)
        logger.info(f"Saved stable model artifact to {stable_path}")

        entry = register_model(
            name=self.model_name,
            artifact_path=stable_path,
            metrics=metrics,
            metadata={"train_rows": len(X_train), "val_rows": len(X_val)},
        )

        # optional MLflow logging (if configured) - left as hook for extension
        meta_path = os.path.join(settings.BASE_DIR, "runs", "training_meta.json")
        write_json(meta_path, {"registry_entry": entry, "metrics": metrics})
        return entry
