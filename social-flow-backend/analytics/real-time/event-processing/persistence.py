import json
import psycopg2
import boto3
from elasticsearch import Elasticsearch
from .config import settings
from .utils import get_logger, retry

logger = get_logger(__name__)


@retry(retries=3, delay=2)
def store_postgres(event: dict) -> None:
    """Persist event in Postgres."""
    conn = psycopg2.connect(settings.postgres_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (event_id, user_id, event_type, value, metadata, ts) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    event["event_id"],
                    event["user_id"],
                    event["event_type"],
                    event["value"],
                    json.dumps(event.get("metadata", {})),
                    event["timestamp"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def store_s3(event: dict) -> None:
    """Persist event in S3 as JSON."""
    s3 = boto3.client("s3")
    key = f"events/{event['timestamp']}/{event['event_id']}.json"
    s3.put_object(Bucket=settings.s3_bucket, Key=key, Body=json.dumps(event).encode("utf-8"))


def store_elasticsearch(event: dict) -> None:
    """Index event in Elasticsearch for querying."""
    es = Elasticsearch(settings.elasticsearch_url)
    es.index(index="events", id=event["event_id"], document=event)
