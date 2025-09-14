# monitoring/logging/analysis/tests/test_correlation_engine.py
import pandas as pd
from monitoring.logging.analysis.correlation_engine import CorrelationEngine

def test_correlation():
    engine = CorrelationEngine()
    logs = [
        {"timestamp": pd.Timestamp("2023-01-01 12:00:00"), "host": "svc1"},
        {"timestamp": pd.Timestamp("2023-01-01 12:00:01"), "host": "svc2"}
    ]
    corr = engine.correlate(logs)
    assert isinstance(corr, list)
