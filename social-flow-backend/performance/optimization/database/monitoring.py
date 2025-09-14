# Monitors database health and performance
import statistics
from collections import defaultdict


class DatabaseMetricsCollector:
    """
    Collects database metrics:
    - Query latency
    - Errors
    - Per-table stats
    """

    def __init__(self):
        self.latencies = []
        self.errors = 0
        self.queries = 0
        self.table_stats = defaultdict(list)

    def record_query(self, table: str, latency: float, success: bool = True):
        self.queries += 1
        if not success:
            self.errors += 1
        self.latencies.append(latency)
        self.table_stats[table].append(latency)

    def get_summary(self):
        if not self.latencies:
            return {"queries": 0}
        return {
            "queries": self.queries,
            "errors": self.errors,
            "error_rate": self.errors / self.queries,
            "avg_latency": statistics.mean(self.latencies),
            "p95_latency": statistics.quantiles(self.latencies, n=100)[94],
        }
