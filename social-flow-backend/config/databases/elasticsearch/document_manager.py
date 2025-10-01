"""CRUD operations for documents, including bulk indexing."""
"""
document_manager.py
-------------------
Handles CRUD operations and bulk indexing for Elasticsearch documents.
"""

from .connection import ElasticsearchClient
from elasticsearch import helpers
import logging

logger = logging.getLogger("DocumentManager")
logger.setLevel(logging.INFO)


class DocumentManager:
    def __init__(self):
        self.client = ElasticsearchClient().get_client()

    def insert_document(self, index: str, doc_id: str, document: dict):
        """Insert or update a document."""
        self.client.index(index=index, id=doc_id, document=document)
        logger.info(f"?? Document {doc_id} indexed in {index}")

    def get_document(self, index: str, doc_id: str):
        """Retrieve document by ID."""
        return self.client.get(index=index, id=doc_id)

    def delete_document(self, index: str, doc_id: str):
        """Delete a document."""
        self.client.delete(index=index, id=doc_id, ignore=[404])
        logger.info(f"?? Document {doc_id} deleted from {index}")

    def bulk_insert(self, index: str, documents: list):
        """Bulk insert documents (efficient)."""
        actions = [{"_index": index, "_id": doc["id"], "_source": doc} for doc in documents]
        helpers.bulk(self.client, actions)
        logger.info(f"? Bulk insert completed for index {index}")
