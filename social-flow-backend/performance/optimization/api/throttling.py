# Controls traffic throttling for clients
import asyncio


class RequestThrottler:
    """
    Request Throttler with Queue Backpressure.

    - Limits concurrent requests.
    - Async-safe using asyncio.Semaphore.
    """

    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process(self, coro):
        async with self.semaphore:
            return await coro
