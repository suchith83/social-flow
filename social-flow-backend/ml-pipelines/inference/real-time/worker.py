# Async background workers
# ================================================================
# File: worker.py
# Purpose: Background async workers (message queue consumers, etc.)
# ================================================================

import asyncio
import logging

logger = logging.getLogger("Worker")


async def message_consumer(queue: asyncio.Queue, handler):
    """Consume incoming messages and run inference"""
    while True:
        msg = await queue.get()
        try:
            result = await handler(msg)
            logger.info(f"Processed message: {result}")
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
        queue.task_done()
