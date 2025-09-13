# Optimizes DB queries
"""
Query Optimizer
Optimizes database queries by caching results and batching.
"""

import time
from collections import defaultdict
from typing import Callable
from .config import CONFIG


class QueryOptimizer:
    def __init__(self):
        self.query_cache = {}
        self.query_batches = defaultdict(list)

    def run_query(self, query: str, exec_fn: Callable):
        """Run or cache a query result."""
        if query in self.query_cache:
            result, expiry = self.query_cache[query]
            if expiry > time.time():
                return result

        result = exec_fn(query)
        self.query_cache[query] = (result, time.time() + CONFIG.query_cache_ttl_sec)
        return result

    def batch_query(self, table: str, query: str, exec_fn: Callable):
        """Batch small queries to reduce DB hits."""
        self.query_batches[table].append(query)
        if len(self.query_batches[table]) >= CONFIG.query_batch_size:
            combined_query = f"SELECT * FROM {table} WHERE id IN ({','.join(self.query_batches[table])})"
            result = exec_fn(combined_query)
            self.query_batches[table].clear()
            return result
        return None
