# Manages database connection pooling
import asyncio
import threading
import queue
import psycopg2
import asyncpg
from typing import Optional


class ThreadedConnectionPool:
    """
    Threaded PostgreSQL Connection Pool (blocking).
    Uses psycopg2 and Queue for pooling.
    """

    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 10):
        self.dsn = dsn
        self.pool = queue.Queue(maxconn)
        for _ in range(minconn):
            self.pool.put(psycopg2.connect(dsn))

    def getconn(self):
        return self.pool.get()

    def putconn(self, conn):
        if conn.closed == 0:
            self.pool.put(conn)

    def closeall(self):
        while not self.pool.empty():
            conn = self.pool.get_nowait()
            conn.close()


class AsyncConnectionPool:
    """
    Async PostgreSQL Connection Pool using asyncpg.
    """

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None

    async def init(self):
        self._pool = await asyncpg.create_pool(
            dsn=self.dsn, min_size=self.min_size, max_size=self.max_size
        )

    async def acquire(self):
        return await self._pool.acquire()

    async def release(self, conn):
        await self._pool.release(conn)

    async def close(self):
        await self._pool.close()
