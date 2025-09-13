"""Efficient data ingestion (batch writes, async writes)."""
"""
write_manager.py
----------------
Handles efficient data ingestion into InfluxDB using batch writes.
"""

from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS, WriteOptions
from .connection import InfluxDBConnection
import logging

logger = logging.getLogger("InfluxDBWriteManager")
logger.setLevel(logging.INFO)


class WriteManager:
    def __init__(self):
        self.client = InfluxDBConnection().get_client()
        self.write_api = self.client.write_api(write_options=WriteOptions(batch_size=500, flush_interval=10_000))
        self.bucket = self.client.buckets_api().find_bucket_by_name("socialflow-metrics").name
        self.org = "socialflow-org"

    def write_point(self, measurement: str, fields: dict, tags: dict = None):
        """Write a single point."""
        point = Point(measurement)
        for k, v in fields.items():
            point = point.field(k, v)
        if tags:
            for k, v in tags.items():
                point = point.tag(k, v)
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        logger.info(f"📊 Point written: {measurement} {fields}")

    def bulk_write(self, points: list):
        """Bulk write multiple points."""
        self.write_api.write(bucket=self.bucket, org=self.org, record=points)
        logger.info(f"⚡ Bulk write completed: {len(points)} points")
