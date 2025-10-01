"""Helper utilities for diagnostics, schema checks, performance testing."""
"""
utils.py
--------
Helper utilities for InfluxDB diagnostics and debugging.
"""

from .connection import InfluxDBConnection


def list_buckets():
    """List all buckets in InfluxDB."""
    client = InfluxDBConnection().get_client()
    return client.buckets_api().find_buckets().buckets


def get_bucket_schema(bucket_name: str):
    """Check bucket schema by sampling data."""
    client = InfluxDBConnection().get_client()
    query = f'from(bucket:"{bucket_name}") |> range(start: -1d) |> limit(n:10)'
    return client.query_api().query(query, org="socialflow-org")
