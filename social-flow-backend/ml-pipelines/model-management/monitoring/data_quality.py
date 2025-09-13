# Data quality checks and validation
"""
data_quality.py
Basic batch data quality checks for model inputs:
 - missing value checks
 - distribution checks
 - cardinality checks
 - schema validation (lightweight)
Designed to be run as part of monitoring pipeline or batch jobs.
"""

import pandas as pd
from typing import Dict, Any
from utils import setup_logger
import numpy as np

logger = setup_logger("DataQuality")


def missing_value_report(df: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
    """
    Returns columns with missing fraction > threshold.
    """
    missing = df.isnull().mean()
    problem = missing[missing > threshold].to_dict()
    logger.info(f"Missing value report: {problem}")
    return {"missing_fraction": missing.to_dict(), "problem_cols": problem}


def distribution_summary(df: pd.DataFrame, numeric_cols: list = None):
    """
    Summary stats for numeric columns: mean, std, min, max, quantiles
    """
    numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[numeric_cols].describe().to_dict()
    logger.info(f"Distribution summary computed for {len(numeric_cols)} columns")
    return desc


def cardinality_checks(df: pd.DataFrame, categorical_cols: list = None, threshold_unique: float = 0.5):
    """
    Flag categorical columns with too high cardinality relative to rows.
    """
    categorical_cols = categorical_cols or df.select_dtypes(include=["object", "category"]).columns.tolist()
    n = len(df)
    issues = {}
    for c in categorical_cols:
        uniq = df[c].nunique()
        if (uniq / max(1, n)) > threshold_unique:
            issues[c] = {"unique": uniq, "frac": uniq / max(1, n)}
    logger.info(f"Cardinality check issues: {issues}")
    return issues
