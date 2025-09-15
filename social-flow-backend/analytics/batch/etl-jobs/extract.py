import psycopg2
import boto3
from kafka import KafkaConsumer
import json
from .config import settings
from .utils import logger, retry


class Extractor:
    """Handles extraction from multiple data sources"""

    @retry(Exception, tries=3)
    def from_postgres(self, query: str):
        """Extract data from PostgreSQL"""
        conn = psycopg2.connect(settings.POSTGRES_URI)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        logger.info(f"Extracted {len(rows)} rows from PostgreSQL")
        return rows

    @retry(Exception, tries=3)
    def from_s3(self, key: str):
        """Extract data from S3"""
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        obj = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
        data = obj["Body"].read().decode("utf-8")
        logger.info(f"Extracted data from S3 key: {key}")
        return json.loads(data)

    def from_kafka(self, timeout_ms=5000, max_messages=100):
        """Consume events from Kafka topic"""
        consumer = KafkaConsumer(
            settings.KAFKA_TOPIC,
            bootstrap_servers=settings.KAFKA_BROKERS,
            auto_offset_reset="earliest",
            consumer_timeout_ms=timeout_ms,
        )
        messages = []
        for i, msg in enumerate(consumer):
            if i >= max_messages:
                break
            messages.append(json.loads(msg.value.decode("utf-8")))
        consumer.close()
        logger.info(f"Consumed {len(messages)} messages from Kafka")
        return messages
