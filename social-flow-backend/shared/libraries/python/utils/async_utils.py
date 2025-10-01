# async_utils.py
import asyncio
from typing import List, Coroutine


async def gather_with_concurrency(limit: int, tasks: List[Coroutine]):
    """
    Run async tasks with concurrency limit.
    """
    semaphore = asyncio.Semaphore(limit)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(t) for t in tasks))
