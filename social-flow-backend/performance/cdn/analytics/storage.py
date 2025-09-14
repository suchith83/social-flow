# Handles storing analytics in DB/warehouse
# performance/cdn/analytics/storage.py
"""
Storage module
==============
Stores processed analytics data into a backend (PostgreSQL/ClickHouse/NoSQL).
This implementation uses async DB writes for scalability.
"""

import asyncio
import asyncpg
from typing import Dict
from .utils import logger

class CDNStorage:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=10)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cdn_metrics (
                    id SERIAL PRIMARY KEY,
                    endpoint TEXT,
                    timestamp TIMESTAMPTZ,
                    latency_ms FLOAT,
                    status_code INT,
                    cache_hit BOOLEAN
                );
            """)
        logger.info("Connected to DB and ensured schema exists.")

    async def store(self, record: Dict):
        """Store a processed record into the database."""
        if not self.pool:
            await self.connect()
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO cdn_metrics (endpoint, timestamp, latency_ms, status_code, cache_hit)
                    VALUES ($1, $2, $3, $4, $5);
                """, record["endpoint"], record["timestamp"], record["latency_ms"],
                     record["status_code"], record["cache_hit"])
        except Exception as e:
            logger.error(f"DB insert failed: {e}")
