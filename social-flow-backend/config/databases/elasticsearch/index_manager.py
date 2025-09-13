"""Create, update, delete indexes with mappings & settings."""
"""
index_manager.py
----------------
Handles index creation, deletion, and updates with mappings and settings.
"""

import logging
from .connection import ElasticsearchClient

logger = logging.getLogger("IndexManager")
logger.setLevel(logging.INFO)


class IndexManager:
    def __init__(self):
        self.client = ElasticsearchClient().get_client()

    def create_index(self, index_name: str, mappings: dict, settings: dict = None):
        """Create index with mappings and settings."""
        if self.client.indices.exists(index=index_name):
            logger.warning(f"Index {index_name} already exists.")
            return
        body = {"mappings": mappings}
        if settings:
            body["settings"] = settings
        self.client.indices.create(index=index_name, body=body)
        logger.info(f"✅ Index {index_name} created successfully.")

    def delete_index(self, index_name: str):
        """Delete index safely."""
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            logger.info(f"🗑 Index {index_name} deleted.")
        else:
            logger.warning(f"Index {index_name} does not exist.")

    def list_indexes(self):
        """List all indexes in the cluster."""
        return self.client.indices.get_alias("*").keys()
