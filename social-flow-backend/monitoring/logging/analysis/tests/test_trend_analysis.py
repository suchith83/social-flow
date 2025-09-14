# Tests for trend_analysis
# monitoring/logging/analysis/tests/test_trend_analysis.py
import pandas as pd
from monitoring.logging.analysis.trend_analysis import TrendAnalyzer

def test_trend_analysis():
    analyzer = TrendAnalyzer()
    logs = [{"timestamp": pd.Timestamp("2023-01-01 12:00:00"), "level": "INFO"}]
    df = analyzer.aggregate(logs)
    assert not df.empty
