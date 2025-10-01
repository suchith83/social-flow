"""Snapshots & restores compatible with S3/GCS/Azure."""
"""
backup_manager.py
-----------------
Handles snapshots & restores for Elasticsearch (supports S3/GCS/Azure).
"""

from .connection import ElasticsearchClient
import logging

logger = logging.getLogger("BackupManager")
logger.setLevel(logging.INFO)


class BackupManager:
    def __init__(self):
        self.client = ElasticsearchClient().get_client()

    def register_repository(self, repo_name: str, repo_type: str, settings: dict):
        """Register snapshot repository (S3/GCS/Azure)."""
        self.client.snapshot.create_repository(
            repository=repo_name,
            body={"type": repo_type, "settings": settings},
            verify=True
        )
        logger.info(f"? Repository {repo_name} registered.")

    def create_snapshot(self, repo_name: str, snapshot_name: str, indices="*"):
        """Create snapshot."""
        self.client.snapshot.create(
            repository=repo_name,
            snapshot=snapshot_name,
            body={"indices": indices, "ignore_unavailable": True, "include_global_state": True}
        )
        logger.info(f"?? Snapshot {snapshot_name} created in {repo_name}")

    def restore_snapshot(self, repo_name: str, snapshot_name: str):
        """Restore snapshot."""
        self.client.snapshot.restore(repository=repo_name, snapshot=snapshot_name)
        logger.info(f"?? Restored snapshot {snapshot_name} from {repo_name}")
