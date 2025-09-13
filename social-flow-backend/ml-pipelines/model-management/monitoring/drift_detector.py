# Detect model/data drift over time
"""
drift_detector.py
A lightweight statistical drift detector for model inputs / features.
Implements:
 - Population Stability Index (PSI)
 - Kolmogorov-Smirnov (KS) test wrapper
 - Simple scoring and thresholding with sliding windows
This is intended as a building block; production deployment should use robust streaming approaches.
"""

import numpy as np
from scipy import stats
import logging
from typing import List, Dict
from utils import setup_logger

logger = setup_logger("DriftDetector")


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index between expected (reference) and actual distributions.
    Lower is better. > 0.2 is often considered significant drift.
    """
    def _buckets(arr, bins):
        return np.histogram(arr, bins=bins)[0].astype(float)

    min_v = min(np.min(expected), np.min(actual))
    max_v = max(np.max(expected), np.max(actual))
    if min_v == max_v:
        return 0.0
    bins = np.linspace(min_v, max_v, buckets + 1)
    e_counts = np.histogram(expected, bins=bins)[0] / (len(expected) + 1e-9)
    a_counts = np.histogram(actual, bins=bins)[0] / (len(actual) + 1e-9)
    # avoid zeros
    e_counts = np.where(e_counts == 0, 1e-6, e_counts)
    a_counts = np.where(a_counts == 0, 1e-6, a_counts)
    psi_vals = (e_counts - a_counts) * np.log(e_counts / a_counts)
    return np.sum(psi_vals)


def ks_test(expected: np.ndarray, actual: np.ndarray):
    """
    Two-sample KS test. Returns statistic and p-value.
    """
    stat, pvalue = stats.ks_2samp(expected, actual)
    return float(stat), float(pvalue)


class DriftDetector:
    """
    Track reference windows and compute drift against recent windows.
    """

    def __init__(self, feature_refs: Dict[str, np.ndarray], psi_threshold: float = 0.2, ks_pvalue_threshold: float = 0.01):
        # feature_refs: dict of feature_name -> numpy array of historical values
        self.refs = feature_refs
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold

    def evaluate(self, feature_name: str, recent_values: np.ndarray):
        if feature_name not in self.refs:
            raise ValueError("No reference for feature")
        ref = self.refs[feature_name]
        score = psi(ref, recent_values)
        stat, p = ks_test(ref, recent_values)
        drift = {
            "feature": feature_name,
            "psi": float(score),
            "ks_stat": float(stat),
            "ks_pvalue": float(p),
            "drift_detected": (score > self.psi_threshold) or (p < self.ks_pvalue_threshold)
        }
        logger.info(f"Drift eval {feature_name}: psi={score:.4f}, ks_stat={stat:.4f}, p={p:.6f}, drift={drift['drift_detected']}")
        return drift
