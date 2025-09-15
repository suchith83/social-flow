"""
Preprocessing Pipeline
- Loads raw features (Parquet / DB)
- Applies the feature engineering pipeline from analytics.predictive.models.feature_engineering
- Persists processed features (train/val/test) to feature store
- Exposes `run()` and `dry_run()` for orchestrator compatibility
"""

from typing import Optional
import os
import pandas as pd
from .config import settings
from .utils import logger, ensure_dir, write_json
from analytics.predictive.models.data_loader import DataLoader
from analytics.predictive.models.feature_engineering import build_feature_pipeline, persist_pipeline
from sklearn.model_selection import train_test_split


class PreprocessingPipeline:
    name = "preprocessing"

    def __init__(
        self,
        feature_table: str = "raw_events",
        target_col: str = "target",
        test_size: float = 0.2,
        val_size: float = 0.1,
        numeric_features: list | None = None,
        categorical_features: list | None = None,
        output_prefix: Optional[str] = None,
    ):
        self.feature_table = feature_table
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.output_prefix = output_prefix or os.path.join(settings.FEATURE_STORE, feature_table)
        ensure_dir(self.output_prefix)
        self.loader = DataLoader()

    def dry_run(self):
        # Check source availability (for local FS we check the parquet file)
        path = f"{self.output_prefix}.parquet"
        logger.info(f"Preprocessing dry-run checking data source: {path}")
        if not os.path.exists(path):
            logger.warning(f"Source data not present at {path}; dry run will proceed but run() may fail.")

    def run(self):
        # Load raw features (prefers local feature store; fallback to SQL)
        local_path = f"{self.output_prefix}.parquet"
        if os.path.exists(local_path):
            df = pd.read_parquet(local_path)
            logger.info(f"Loaded raw features from {local_path} shape={df.shape}")
        else:
            # fallback: fetch from Snowflake query - user must supply SNOWFLAKE_URI in settings
            if not settings.SNOWFLAKE_URI:
                raise RuntimeError("No local feature file and no SNOWFLAKE_URI configured")
            sql = f"SELECT * FROM {self.feature_table}"
            df = self.loader.fetch_from_sql(sql)
            logger.info(f"Loaded raw features from DW table {self.feature_table} shape={df.shape}")

        # basic sanity checks
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in data")

        # split train/val/test
        test_pct = self.test_size
        val_pct = self.val_size
        # compute splits: we first split off test, then val from train
        train_val_df, test_df = train_test_split(df, test_size=test_pct, random_state=settings.RANDOM_SEED)
        train_df, val_df = train_test_split(train_val_df, test_size=val_pct / (1 - test_pct), random_state=settings.RANDOM_SEED)

        logger.info(f"Split data: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")

        # build and fit feature pipeline on train dataset
        X_train = train_df.drop(columns=[self.target_col])
        feat_pipeline = build_feature_pipeline(self.numeric_features, self.categorical_features)
        feat_pipeline.fit(X_train)
        persist_pipeline(feat_pipeline, name="feature_pipeline.pkl")

        # transform and persist processed datasets
        X_train_trans = feat_pipeline.transform(X_train)
        train_out = os.path.join(settings.FEATURE_STORE, f"{self.feature_table}_train.parquet")
        pd.DataFrame(X_train_trans).assign(**{self.target_col: train_df[self.target_col].values}).to_parquet(train_out, index=False)
        logger.info(f"Wrote train features to {train_out}")

        X_val_trans = feat_pipeline.transform(val_df.drop(columns=[self.target_col]))
        val_out = os.path.join(settings.FEATURE_STORE, f"{self.feature_table}_val.parquet")
        pd.DataFrame(X_val_trans).assign(**{self.target_col: val_df[self.target_col].values}).to_parquet(val_out, index=False)
        logger.info(f"Wrote val features to {val_out}")

        X_test_trans = feat_pipeline.transform(test_df.drop(columns=[self.target_col]))
        test_out = os.path.join(settings.FEATURE_STORE, f"{self.feature_table}_test.parquet")
        pd.DataFrame(X_test_trans).assign(**{self.target_col: test_df[self.target_col].values}).to_parquet(test_out, index=False)
        logger.info(f"Wrote test features to {test_out}")

        # metadata
        meta = {
            "train_path": train_out,
            "val_path": val_out,
            "test_path": test_out,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        }
        meta_path = os.path.join(settings.BASE_DIR, "runs", "preprocessing_meta.json")
        write_json(meta_path, meta)
        logger.info(f"Preprocessing metadata written to {meta_path}")
        return meta
