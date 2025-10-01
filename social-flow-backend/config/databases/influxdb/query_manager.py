"""Advanced queries using Flux and SQL-like syntax."""
"""
query_manager.py
----------------
Executes advanced Flux queries and SQL-like queries on InfluxDB.
"""

from .connection import InfluxDBConnection
import logging

logger = logging.getLogger("InfluxDBQueryManager")
logger.setLevel(logging.INFO)


class QueryManager:
    def __init__(self):
        self.client = InfluxDBConnection().get_client()
        self.query_api = self.client.query_api()
        self.org = "socialflow-org"

    def flux_query(self, query: str):
        """Execute a Flux query."""
        result = self.query_api.query(org=self.org, query=query)
        logger.info(f"?? Executed Flux query: {query}")
        return result

    def sql_query(self, query: str):
        """Execute SQL query (InfluxDB v2 supports SQL-like syntax)."""
        result = self.query_api.query(query, org=self.org, dialect="sql")
        logger.info(f"?? Executed SQL query: {query}")
        return result
