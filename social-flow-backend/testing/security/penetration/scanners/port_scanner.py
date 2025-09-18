# =========================
# File: testing/security/penetration/scanners/port_scanner.py
# =========================
"""
Asynchronous TCP port scanner (connect scan).
- Non-exploitative: only attempts TCP connect and records success/failure.
- Respects concurrency and per-target rate limiting.
"""

import socket
import asyncio
import time
from ..utils.logger import get_logger
from ..utils.network import RateLimiter

logger = get_logger("PortScanner")

async def _try_connect(host: str, port: int, timeout: float):
    loop = asyncio.get_running_loop()
    try:
        fut = loop.run_in_executor(None, lambda: socket.create_connection((host, port), timeout))
        sock = await asyncio.wait_for(fut, timeout=timeout + 0.5)
        sock.close()
        return True
    except Exception:
        return False

async def _scan_target(host: str, ports: list, timeout: float, limiter: RateLimiter, concurrency_sem: asyncio.Semaphore):
    results = {}
    async with concurrency_sem:
        for port in ports:
            # rate-limit across calls
            while not limiter.acquire():
                await asyncio.sleep(0.1)
            ok = await _try_connect(host, port, timeout)
            results[port] = ok
    return results

def scan(host: str, ports: list, timeout: float = 1.0, concurrency: int = 50, max_ops_per_minute: int = 1000):
    """
    Public sync wrapper around the async scanner. Returns dict {port: bool}.
    """
    logger.info(f"Starting port scan of {host} ports={ports} timeout={timeout} concurrency={concurrency}")
    limiter = RateLimiter(max_ops_per_minute, period_seconds=60.0)
    async def runner():
        sem = asyncio.Semaphore(concurrency)
        return await _scan_target(host, ports, timeout, limiter, sem)

    results = asyncio.run(runner())
    logger.info(f"Finished port scan of {host}")
    return results
