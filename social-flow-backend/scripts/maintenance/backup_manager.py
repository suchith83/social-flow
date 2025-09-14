# scripts/maintenance/backup_manager.py
import os
import logging
import tarfile
import tempfile
import boto3
from datetime import datetime, timedelta
from typing import Dict, Any, List

from .utils import ensure_dir, now_ts, run_cmd

logger = logging.getLogger("maintenance.backup")

class BackupManager:
    """
    Creates backups for directories or databases and uploads to S3 (optional).
    Safe, idempotent, and supports retention policy.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("maintenance", {}).get("backup", {})
        self.s3_bucket = self.cfg.get("s3_bucket")
        self.retention_days = int(self.cfg.get("retention_days", 30))
        self.local_repo = self.cfg.get("local_repo", "/var/backups/socialflow")
        ensure_dir(self.local_repo)

        # initialize boto3 if s3 configured
        self.s3_client = None
        if self.s3_bucket:
            try:
                self.s3_client = boto3.client("s3")
            except Exception:
                logger.exception("Could not initialize boto3 S3 client; S3 uploads disabled")

    def _tar_paths(self, name: str, paths: List[str]) -> str:
        ts = now_ts()
        out_name = f"{name}-{ts}.tar.gz"
        out_path = os.path.join(self.local_repo, out_name)
        logger.info("Creating tarball %s for paths: %s", out_path, paths)
        with tarfile.open(out_path, "w:gz") as tar:
            for p in paths:
                if os.path.exists(p):
                    tar.add(p, arcname=os.path.basename(p))
                else:
                    logger.warning("Path does not exist; skipping: %s", p)
        return out_path

    def backup_directories(self, name: str, paths: List[str], upload: bool = True) -> str:
        tarball = self._tar_paths(name, paths)
        if upload and self.s3_client:
            key = os.path.basename(tarball)
            try:
                logger.info("Uploading %s to s3://%s/%s", tarball, self.s3_bucket, key)
                self.s3_client.upload_file(tarball, self.s3_bucket, key)
            except Exception:
                logger.exception("S3 upload failed")
        return tarball

    def backup_postgres(self, name: str, dsn: str, upload: bool = True) -> str:
        """
        Uses pg_dump to create a logical backup. DSN can be connection string or PG env vars must be set.
        """
        ts = now_ts()
        outfile = os.path.join(self.local_repo, f"{name}-{ts}.sql.gz")
        logger.info("Creating Postgres dump to %s", outfile)
        # Using shell pipeline: pg_dump | gzip > outfile
        cmd = ["bash", "-lc", f"pg_dump \"{dsn}\" | gzip > \"{outfile}\""]
        run_cmd(cmd)
        if upload and self.s3_client:
            key = os.path.basename(outfile)
            try:
                logger.info("Uploading %s to s3://%s/%s", outfile, self.s3_bucket, key)
                self.s3_client.upload_file(outfile, self.s3_bucket, key)
            except Exception:
                logger.exception("S3 upload failed for postgres dump")
        return outfile

    def prune_local(self):
        """
        Remove local backup files older than retention_days
        """
        cutoff = datetime.utcnow() - timedelta(days=int(self.retention_days))
        logger.info("Pruning local backups older than %s", cutoff.isoformat())
        for f in os.listdir(self.local_repo):
            path = os.path.join(self.local_repo, f)
            try:
                mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
                if mtime < cutoff:
                    logger.info("Removing old backup: %s", path)
                    os.remove(path)
            except Exception:
                logger.exception("Failed to remove file: %s", path)
