# Publishes results back to a topic
# ================================================================
# File: producer.py
# Purpose: Publishes inference results to Kafka/Pulsar
# ================================================================

import asyncio
import logging
from aiokafka import AIOKafkaProducer

logger = logging.getLogger("StreamProducer")


class StreamProducer:
    def __init__(self, config: dict):
        self.config = config
        self.producer = None

    async def start(self):
        self.producer = AIOKafkaProducer(
            loop=asyncio.get_event_loop(),
            bootstrap_servers=self.config["bootstrap_servers"],
        )
        await self.producer.start()

    async def send(self, message: dict):
        if not self.producer:
            await self.start()
        topic = self.config["topic"]
        await self.producer.send_and_wait(topic, str(message).encode("utf-8"))
        logger.info(f"📤 Published result to {topic}: {message}")
