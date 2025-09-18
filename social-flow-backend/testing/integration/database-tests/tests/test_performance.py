"""
Micro-benchmark: measure simple insert throughput. Marked slow.

This test is intended to give an idea of relative performance and spot regressions.
It is intentionally small but configurable.
"""

import pytest
import time
from db.sync_client import engine
from db.models import User
from sqlalchemy import insert

@pytest.mark.integration
@pytest.mark.slow
def test_bulk_insert_performance():
    n = 500
    start = time.time()
    with engine.begin() as conn:
        stmt = insert(User)
        for i in range(n):
            conn.execute(stmt.values(username=f"perf_{i}_{int(time.time())}", hashed_password="x"))
    elapsed = time.time() - start
    rate = n / elapsed
    # basic assertion: ensure rate is > 10 inserts/sec (tunable)
    assert rate > 10, f"Expected >10 inserts/sec; got {rate:.2f}"
