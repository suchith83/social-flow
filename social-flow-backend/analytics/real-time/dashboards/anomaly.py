import statistics
from typing import List
from .config import settings
from .utils import get_logger

logger = get_logger(__name__)


class ZScoreAnomalyDetector:
    """Detects anomalies in a data stream using Z-score."""

    def __init__(self, threshold: float = None):
        self.threshold = threshold or settings.anomaly_threshold

    def detect(self, data: List[float]) -> bool:
        """Returns True if the latest value is anomalous."""
        if len(data) < 2:
            return False
        mean = statistics.mean(data)
        stdev = statistics.pstdev(data)
        if stdev == 0:
            return False
        z_score = (data[-1] - mean) / stdev
        is_anomaly = abs(z_score) > self.threshold
        if is_anomaly:
            logger.warning(f"Anomaly detected: value={data[-1]}, z={z_score:.2f}")
        return is_anomaly
