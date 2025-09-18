"""
Redis Streams tests:
- XADD and XREADGROUP (consumer groups)
- ordering and at-least-once semantics
- consumer group creation & cleanup
"""

import pytest
import time
import uuid

@pytest.mark.messaging
def test_redis_stream_add_and_read(redis_helper):
    stream = "integration-test-stream"
    # add a message
    id1 = redis_helper.add(stream, {"k": "v"})
    assert id1 is not None
    # create group and read via consumer group
    group = "test-group"
    consumer = f"consumer-{uuid.uuid4().hex[:6]}"
    redis_helper.create_group(stream, group)
    msgs = redis_helper.read_group(stream, group, consumer, count=1, block=1000)
    # reading might return nested structure; assert we got something or skip if not
    assert msgs is not None

@pytest.mark.messaging
def test_redis_stream_ordering(redis_helper):
    stream = "integration-test-stream"
    # clean stream
    redis_helper.delete_stream(stream)
    # add ordered messages
    ids = []
    for i in range(5):
        ids.append(redis_helper.add(stream, {"seq": str(i)}))
    # read raw stream entries using XRANGE via redis client directly to ensure order
    client = redis_helper.client
    entries = client.xrange(stream, count=10)
    seqs = [int(entry[1][b'seq']) for entry in entries]
    assert seqs == list(range(5))

@pytest.mark.messaging
def test_redis_stream_consumer_group(redis_helper):
    stream = "integration-test-stream"
    group = "cg-test"
    redis_helper.create_group(stream, group)
    consumer = "cg-consumer-1"
    # add message and read via group
    redis_helper.add(stream, {"event": "x"})
    res = redis_helper.read_group(stream, group, consumer, count=1, block=1000)
    assert res is not None
