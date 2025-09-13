# Kafka/Pulsar consumer logic
# ================================================================
# File: consumer.py
# Purpose: Consume streaming messages (Kafka/Pulsar)
# ================================================================

import asyncio
import logging
from aiokafka import AIOKafkaConsumer

logger = logging.getLogger("StreamConsumer")


class StreamConsumer:
    def __init__(self, config: dict, worker, producer, monitoring):
        self.config = config
        self.worker = worker
        self.producer = producer
        self.monitoring = monitoring
        self.consumer = None

    async def start(self):
        topic = self.config["topic"]
        bootstrap = self.config["bootstrap_servers"]

        self.consumer = AIOKafkaConsumer(
            topic,
            loop=asyncio.get_event_loop(),
            bootstrap_servers=bootstrap,
            group_id=self.config.get("group_id", "inference-group"),
            enable_auto_commit=True,
        )

        await self.consumer.start()
        try:
            logger.info(f"✅ Subscribed to topic {topic}")
            async for msg in self.consumer:
                try:
                    payload = msg.value.decode("utf-8")
                    result = await self.worker.run(payload)
                    await self.producer.send(result)
                    self.monitoring.log_inference(success=True)
                except Exception as e:
                    logger.error(f"Inference failed: {e}")
                    self.monitoring.log_inference(success=False)
        finally:
            await self.consumer.stop()
