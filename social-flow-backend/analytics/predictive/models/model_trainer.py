"""
Model training:
 - Supports XGBoost and sklearn regressors/classifiers
 - Cross-validation, randomized search (scikit-learn)
 - Produces a trained pipeline artifact (feature pipeline + model)
"""

from typing import Any, Dict, Tuple
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .feature_engineering import build_feature_pipeline, persist_pipeline
from .utils import logger, save_pickle, timestamp
from .config import settings
import joblib


class Trainer:
    def __init__(self, model_dir: str = settings.MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train_xgb_regressor(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        numeric_features=None,
        categorical_features=None,
        param_distributions: Dict | None = None,
        n_iter: int = 30,
        cv_splits: int = 3,
    ) -> Tuple[Pipeline, Dict]:
        """
        Train an XGBoost regressor wrapped in a sklearn Pipeline:
          pipeline = Pipeline([('features', feature_pipeline), ('model', xgb)])
        Returns (pipeline, metrics)
        """
        # Build feature pipeline
        feat_pipeline = build_feature_pipeline(numeric_features or [], categorical_features or [])
        xgb = XGBRegressor(
            n_jobs=settings.N_JOBS,
            random_state=settings.RANDOM_SEED,
            objective="reg:squarederror",
            verbosity=0,
        )

        pipeline = Pipeline([("features", feat_pipeline), ("model", xgb)])

        # Parameter distribution for randomized search
        default_params = {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [3, 5, 7, 9],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.5, 0.7, 1.0],
            "model__reg_alpha": [0, 0.1, 0.5],
            "model__reg_lambda": [1, 2, 5],
        }
        param_distributions = param_distributions or default_params

        # Use time-series-aware split if appropriate
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            verbose=1,
            n_jobs=settings.N_JOBS,
            random_state=settings.RANDOM_SEED,
        )

        logger.info("Starting randomized search CV for XGBoost regressor")
        search.fit(X_train, y_train)

        logger.info(f"Best params: {search.best_params_}")
        best_pipeline: Pipeline = search.best_estimator_

        # Persist pipeline and feature pipeline separately for re-use
        model_name = f"{settings.DEFAULT_MODEL_NAME}_{timestamp()}.pkl"
        model_path = os.path.join(self.model_dir, model_name)
        save_pickle(best_pipeline, model_path)
        logger.info(f"Saved trained model to {model_path}")

        # Persist just the feature pipeline if needed
        persist_pipeline(best_pipeline.named_steps["features"], name="feature_pipeline.pkl")

        # Evaluate if val set provided
        metrics = {}
        if X_val is not None and y_val is not None:
            preds = best_pipeline.predict(X_val)
            metrics = {
                "mae": float(mean_absolute_error(y_val, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
                "r2": float(r2_score(y_val, preds)),
            }
            logger.info(f"Validation metrics: {metrics}")

        return best_pipeline, metrics
