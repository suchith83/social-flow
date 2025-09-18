"""
RabbitMQ helper based on pika (sync). Provides:
- connect / close
- publish to queue / consume (basic_get & callback)
- declare queue, purge, and dead-letter wiring helpers
"""

import json
import logging
import pika
from pika.exceptions import AMQPConnectionError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RabbitMQHelper:
    def __init__(self, amqp_url: str):
        self.url = amqp_url
        self.conn: pika.BlockingConnection | None = None
        self.channel: pika.adapters.blocking_connection.BlockingChannel | None = None

    def connect(self):
        params = pika.URLParameters(self.url)
        self.conn = pika.BlockingConnection(params)
        self.channel = self.conn.channel()

    def declare_queue(self, queue_name: str, durable: bool = True, dlx: str | None = None):
        args = {}
        if dlx:
            args['x-dead-letter-exchange'] = dlx
        self.channel.queue_declare(queue=queue_name, durable=durable, arguments=args)

    def purge_queue(self, queue_name: str):
        self.channel.queue_purge(queue=queue_name)

    def publish(self, queue_name: str, body: dict, headers: dict | None = None):
        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(body),
            properties=pika.BasicProperties(
                delivery_mode=2,  # persistent
                headers=headers or {}
            )
        )

    def get_message(self, queue_name: str, auto_ack: bool = True, timeout: int = 5):
        # basic_get is polling; use for tests
        end = time.time() + timeout
        while time.time() < end:
            method_frame, properties, body = self.channel.basic_get(queue=queue_name, auto_ack=auto_ack)
            if method_frame:
                try:
                    return json.loads(body)
                except Exception:
                    return body
            time.sleep(0.1)
        return None

    def close(self):
        try:
            if self.channel:
                self.channel.close()
            if self.conn:
                self.conn.close()
        except Exception:
            pass
