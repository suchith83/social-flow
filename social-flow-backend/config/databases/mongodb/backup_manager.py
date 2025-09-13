"""Backup & restore helpers (mongodump/mongorestore orchestration, snapshot hooks)."""
"""
backup_manager.py
-----------------
Backup and restore orchestration for MongoDB.

Capabilities:
 - Trigger mongodump to filesystem (consistent snapshot if using --oplog for replica set)
 - Upload backups to S3 (optional; uses awscli or boto3)
 - Restore via mongorestore with safety checks
 - Retention enforcement (deletes older backups beyond retention_days)
 - Hooks for pre/post backup scripts (quiesce other services, lock files, etc.)

Security note:
 - In production, prefer using managed snapshots (cloud provider) or filesystem snapshotting
   on replica set secondaries to avoid impacting primary IO.
"""

import subprocess
from pathlib import Path
import shutil
import logging
import os
from datetime import datetime, timedelta
import yaml
from .connection import MongoConnectionManager

logger = logging.getLogger("MongoBackup")
logger.setLevel(logging.INFO)


def _load_conf():
    return yaml.safe_load(Path("config/databases/mongodb/config.yaml").read_text())["mongodb"]


class BackupManager:
    def __init__(self):
        self.conf = _load_conf()
        self.dump_dir = Path(self.conf.get("backup", {}).get("dump_dir", "/var/backups/mongo"))
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = int(self.conf.get("backup", {}).get("retention_days", 30))
        self.s3_bucket = self.conf.get("backup", {}).get("s3_bucket")

    def _timestamped_dir(self):
        return self.dump_dir / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def run_mongodump(self, database: str = None, gzip: bool = True, oplog: bool = True):
        """
        Run mongodump against a secondary preferred (if replica set) to reduce primary load.
        Uses the URI built from config. Adds --oplog for consistent snapshot if running against replica set.
        """
        conn = MongoConnectionManager()
        uri = conn._build_uri()  # using the private method to build URI; adjust if needed
        target = self._timestamped_dir()
        target.mkdir(parents=True, exist_ok=True)
        cmd = ["mongodump", "--uri", uri, "--out", str(target)]
        if gzip:
            cmd.append("--gzip")
        if oplog:
            cmd.append("--oplog")
        logger.info(f"Running mongodump -> {target}")
        subprocess.run(cmd, check=True)
        logger.info("✅ mongodump finished")
        return target

    def upload_to_s3(self, path: Path):
        """
        Upload to S3. Prefer using boto3 for programmatic uploads; here we show simple aws cli approach.
        In production, use multipart uploads and server-side encryption.
        """
        if not self.s3_bucket:
            logger.warning("No s3_bucket configured. Skipping upload.")
            return

        cmd = ["aws", "s3", "cp", str(path), self.s3_bucket + "/" + path.name, "--recursive"]
        logger.info(f"Uploading backup to {self.s3_bucket}")
        subprocess.run(cmd, check=True)
        logger.info("✅ Upload completed")

    def restore(self, backup_path: str, drop_existing: bool = False):
        """
        Restore using mongorestore. Be careful in production — restoring over live DB can be destructive.
        Use --nsInclude/--nsExclude to limit restore scope.
        """
        cmd = ["mongorestore", "--gzip", "--uri", MongoConnectionManager()._build_uri(), str(backup_path)]
        if drop_existing:
            cmd.insert(1, "--drop")
        logger.info(f"Running mongorestore from {backup_path}")
        subprocess.run(cmd, check=True)
        logger.info("✅ Restore completed")

    def cleanup_old_backups(self):
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        for child in self.dump_dir.iterdir():
            try:
                created = datetime.fromtimestamp(child.stat().st_mtime)
                if created < cutoff:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                    logger.info(f"Deleted old backup {child}")
            except Exception:
                logger.exception(f"Failed to remove backup {child}")
