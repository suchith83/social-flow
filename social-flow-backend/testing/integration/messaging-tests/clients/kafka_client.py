"""
Kafka helpers:
- AsyncKafkaHelper: uses aiokafka for async produce/consume
- SyncKafkaHelper: uses kafka-python for simple sync produce/consume (useful in sync tests)
"""

import asyncio
import json
import logging
from typing import Callable, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------- async helper -----------------
class AsyncKafkaHelper:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str = "test-group"):
        self.bootstrap = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._task = None
        self._running = False
        self._messages = asyncio.Queue()

    async def start(self):
        # init producer & consumer
        self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap)
        await self._producer.start()
        self._consumer = AIOKafkaConsumer(self.topic, bootstrap_servers=self.bootstrap, group_id=self.group_id,
                                          auto_offset_reset="earliest", enable_auto_commit=True)
        await self._consumer.start()
        self._running = True
        # spawn consumer task
        self._task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        try:
            async for msg in self._consumer:
                # put raw message
                await self._messages.put(msg)
        except Exception as e:
            logger.exception("Kafka consume loop stopped: %s", e)

    async def produce(self, key: Optional[bytes], value: dict, headers: Optional[List[tuple]] = None):
        if not self._producer:
            raise RuntimeError("Producer not started")
        data = json.dumps(value).encode("utf-8")
        await self._producer.send_and_wait(self.topic, value=data, key=key, headers=headers or [])

    async def get_message(self, timeout: float = 5.0):
        try:
            return await asyncio.wait_for(self._messages.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def stop(self):
        self._running = False
        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            await self._producer.stop()
        if self._task:
            self._task.cancel()
            with suppress := contextlib.suppress(Exception):
                await self._task

# ----------------- sync helper -----------------
class SyncKafkaHelper:
    def __init__(self, bootstrap_servers: str, topic: str):
        self.bootstrap = bootstrap_servers
        self.topic = topic
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None

    def start(self):
        self.producer = KafkaProducer(bootstrap_servers=self.bootstrap, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        # consumer not started until get_message is called.

    def produce(self, value: dict, key: Optional[bytes] = None):
        if not self.producer:
            raise RuntimeError("Producer not started")
        fut = self.producer.send(self.topic, value=value, key=key)
        fut.get(timeout=10)

    def get_message(self, timeout: int = 5):
        if not self.consumer:
            self.consumer = KafkaConsumer(self.topic, bootstrap_servers=self.bootstrap, auto_offset_reset='earliest', consumer_timeout_ms=timeout * 1000)
        for msg in self.consumer:
            return msg
        return None

    def stop(self):
        if self.producer:
            self.producer.flush()
            self.producer.close()
        if self.consumer:
            self.consumer.close()
