"""
Redis Streams helper:
- uses redis-py for sync operations and aioredis for async if desired
- helpers: XADD, XREADGROUP, create consumer group, delete group, purge stream
"""

import time
import json
import redis
from redis.exceptions import ResponseError

class RedisStreamsHelper:
    def __init__(self, redis_url: str):
        self.url = redis_url
        self.client: redis.Redis | None = None

    def connect(self):
        self.client = redis.from_url(self.url, decode_responses=False)

    def close(self):
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass

    def ensure_stream(self, stream: str):
        # XADD a dummy message then delete the stream key (keeps stream metadata)
        self.client.xadd(stream, {"hello": "1"})
        # trim to zero - keep stream key
        try:
            self.client.xtrim(stream, 0)
        except ResponseError:
            pass

    def add(self, stream: str, message: dict):
        # redis expects bytes; convert values to bytes
        kv = {k: (v.encode() if isinstance(v, str) else json.dumps(v).encode()) for k, v in message.items()}
        return self.client.xadd(stream, kv)

    def create_group(self, stream: str, group: str):
        try:
            self.client.xgroup_create(stream, group, id="$", mkstream=True)
        except ResponseError:
            # group may already exist
            pass

    def read_group(self, stream: str, group: str, consumer: str, count: int = 1, block: int = 1000):
        # returns list of (stream, [(id, {k: v})])
        return self.client.xreadgroup(groupname=group, consumername=consumer, streams={stream: ">"}, count=count, block=block)

    def delete_stream(self, stream: str):
        self.client.delete(stream)
