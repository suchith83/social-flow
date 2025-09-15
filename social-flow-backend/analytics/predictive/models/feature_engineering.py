"""
Feature engineering module.
- Build sklearn Pipelines for numerical and categorical processing.
- Provide fit/transform helpers for production (persisted transformers).
"""

from typing import Tuple, Iterable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from .utils import logger, save_pickle, load_pickle
from .config import settings
import os


class DateTimeFeatures(TransformerMixin, BaseEstimator):
    """Extracts simple datetime features from a timestamp column."""

    def __init__(self, ts_col: str = "event_ts"):
        self.ts_col = ts_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.ts_col not in X.columns:
            logger.warning(f"Timestamp column {self.ts_col} not present, skipping DateTimeFeatures.")
            return X
        ts = pd.to_datetime(X[self.ts_col])
        X["hour"] = ts.dt.hour
        X["dayofweek"] = ts.dt.dayofweek
        X["is_weekend"] = ts.dt.dayofweek >= 5
        return X


def build_feature_pipeline(
    numeric_features: Iterable[str], categorical_features: Iterable[str], drop_low_variance: bool = True
) -> Pipeline:
    """
    Build and return a preprocessing pipeline and the list of final feature column names.
    Persist the pipeline to models/feature_pipeline.pkl if desired.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    steps = [("preprocessor", preprocessor)]
    if drop_low_variance:
        steps.append(("variance", VarianceThreshold(threshold=0.0)))

    pipeline = Pipeline(steps=steps)
    logger.info("Built feature engineering pipeline")
    return pipeline


def persist_pipeline(pipeline: Pipeline, name: str = "feature_pipeline.pkl"):
    path = os.path.join(settings.MODEL_DIR, name)
    save_pickle(pipeline, path)


def load_pipeline(name: str = "feature_pipeline.pkl") -> Pipeline:
    path = os.path.join(settings.MODEL_DIR, name)
    return load_pickle(path)
