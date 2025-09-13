"""Create, ensure, or drop indexes using best practices for background builds."""
"""
index_manager.py
----------------
Helpers to create indexes with best practices:
 - Create indexes in background where supported
 - Use sparse/partial indexes where beneficial
 - Provide idempotent ensure_index functions
 - Support TTL indexes for data expiration
"""

import logging
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import OperationFailure
from .connection import MongoConnectionManager

logger = logging.getLogger("MongoIndexManager")
logger.setLevel(logging.INFO)


class IndexManager:
    def __init__(self):
        conn = MongoConnectionManager()
        self.db = conn.get_database()

    def ensure_index(self, collection_name: str, keys, name: str = None, background: bool = True,
                     unique: bool = False, partial_filter: dict = None, expire_after_seconds: int = None):
        """
        Ensure index exists with the given specification.
        `keys` can be list of tuples like [("field", ASCENDING), ...] or dict
        """
        collection = self.db[collection_name]
        index_options = {"background": background, "unique": unique}
        if partial_filter:
            index_options["partialFilterExpression"] = partial_filter
        if expire_after_seconds is not None:
            index_options["expireAfterSeconds"] = expire_after_seconds

        try:
            idx_name = collection.create_index(keys, name=name, **index_options)
            logger.info(f"Index created/ensured: {collection_name}.{idx_name}")
            return idx_name
        except OperationFailure as e:
            logger.exception(f"Failed creating index on {collection_name}: {e}")
            raise

    def list_indexes(self, collection_name: str):
        return list(self.db[collection_name].list_indexes())

    def drop_index(self, collection_name: str, index_name: str):
        try:
            self.db[collection_name].drop_index(index_name)
            logger.info(f"Dropped index {index_name} on {collection_name}")
        except Exception:
            logger.exception(f"Failed to drop index {index_name} on {collection_name}")
            raise
