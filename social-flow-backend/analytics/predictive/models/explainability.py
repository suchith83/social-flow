"""
Explainability helpers:
 - SHAP integration for tree-based models (XGBoost)
 - Produce local explanations and global feature importance
 Note: SHAP can be heavy; in production you may sample data for explanation jobs.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
from .utils import logger, save_pickle, ensure_dir
from .config import settings
import os

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False
    logger.warning("SHAP library not available; explainability features are disabled.")


def compute_shap_for_model(model_pipeline: Any, X_sample: pd.DataFrame, out_dir: str, nsamples: int = 100):
    """
    Compute SHAP values for a tree-based model wrapped in a pipeline.
    Steps:
      - Extract the final estimator (expects pipeline.named_steps['model'] present)
      - Use TreeExplainer for XGBoost/Tree models
    """
    if not _HAS_SHAP:
        raise RuntimeError("SHAP not installed in environment")

    ensure_dir(out_dir)
    # Extract last estimator
    if hasattr(model_pipeline, "named_steps") and "model" in model_pipeline.named_steps:
        estimator = model_pipeline.named_steps["model"]
    else:
        raise RuntimeError("Pipeline does not expose a 'model' step for SHAP extraction.")

    # Transform features using pipeline.feature transforms
    X_transformed = model_pipeline.named_steps["features"].transform(X_sample)
    logger.info(f"Computing SHAP values on {X_transformed.shape[0]} samples (nsamples={nsamples})")

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_transformed, check_additivity=False)
    # Save shap values and summary
    out_shap_path = os.path.join(out_dir, "shap_values.pkl")
    save_pickle(shap_values, out_shap_path)

    # Compute global importance (mean absolute)
    importance = np.abs(shap_values).mean(axis=0).tolist()
    feature_names = getattr(model_pipeline.named_steps["features"], "get_feature_names_out", lambda: None)()
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    importance_path = os.path.join(out_dir, "shap_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved SHAP importance to {importance_path}")
    return {"shap_path": out_shap_path, "importance_csv": importance_path}
