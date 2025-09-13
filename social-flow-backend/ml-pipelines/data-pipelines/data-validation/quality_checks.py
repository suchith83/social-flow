# Nulls, duplicates, outliers, ranges
# quality_checks.py
import pandas as pd
from typing import Dict, Any
from .utils import logger, timed

class QualityChecker:
    """
    Runs common data quality checks such as missing values, duplicates, and range checks.
    """

    @timed
    def check_missing(self, df: pd.DataFrame) -> Dict[str, Any]:
        missing = df.isnull().sum().to_dict()
        return {"missing_counts": missing}

    @timed
    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        duplicate_count = df.duplicated().sum()
        return {"duplicate_rows": duplicate_count}

    @timed
    def check_ranges(self, df: pd.DataFrame, ranges: Dict[str, tuple]) -> Dict[str, Any]:
        """Check if numeric values fall within given ranges."""
        results = {}
        for col, (min_val, max_val) in ranges.items():
            violations = df[(df[col] < min_val) | (df[col] > max_val)]
            results[col] = {"violations": len(violations), "indices": violations.index.tolist()}
        return results
