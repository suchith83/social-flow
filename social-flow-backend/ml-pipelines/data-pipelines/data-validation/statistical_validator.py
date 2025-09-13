# Checks distributions, drift, anomalies
# statistical_validator.py
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any
from .utils import logger, timed

class StatisticalValidator:
    """
    Detects statistical anomalies like data drift, outliers, and distribution mismatches.
    """

    def __init__(self, reference_df: pd.DataFrame):
        self.reference_df = reference_df

    @timed
    def detect_drift(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for distribution drift."""
        drift_results = {}
        for col in self.reference_df.columns:
            if pd.api.types.is_numeric_dtype(self.reference_df[col]):
                stat, p_value = ks_2samp(
                    self.reference_df[col].dropna(),
                    new_df[col].dropna()
                )
                drift_results[col] = {"ks_stat": stat, "p_value": p_value}
        return drift_results

    @timed
    def detect_outliers(self, df: pd.DataFrame, z_thresh: float = 3.0) -> Dict[str, Any]:
        """Z-score based outlier detection."""
        outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            outlier_indices = df.index[np.abs(z_scores) > z_thresh].tolist()
            outliers[col] = {"count": len(outlier_indices), "indices": outlier_indices}
        return outliers
