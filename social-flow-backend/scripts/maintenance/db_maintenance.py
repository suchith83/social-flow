# scripts/maintenance/db_maintenance.py
import logging
from typing import Dict, Any
from .utils import run_cmd

logger = logging.getLogger("maintenance.db")

class DBMaintenance:
    """
    Handles DB-specific maintenance tasks: vacuum/optimize, index rebuilds, migrations.
    Works with PostgreSQL as default; extendable for others.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("maintenance", {}).get("db", {})
        self.type = self.cfg.get("type", "postgres")
        self.dsn = self.cfg.get("dsn")  # e.g. postgres connection string or environment variables
        self.vacuum = bool(self.cfg.get("vacuum", True))
        self.vacuum_full = bool(self.cfg.get("vacuum_full", False))
        self.migration_cmd = self.cfg.get("migration_cmd")  # e.g. "alembic upgrade head"

    def run_vacuum(self):
        if self.type != "postgres":
            logger.info("Vacuum not implemented for DB type %s", self.type)
            return
        if not self.vacuum:
            logger.info("Vacuum disabled in config")
            return

        if self.vacuum_full:
            cmd = ["bash", "-lc", f"psql \"{self.dsn}\" -c \"VACUUM FULL;\""]
        else:
            cmd = ["bash", "-lc", f"psql \"{self.dsn}\" -c \"VACUUM VERBOSE ANALYZE;\""]
        logger.info("Running vacuum (type=%s, full=%s)", self.type, self.vacuum_full)
        run_cmd(cmd)

    def rebuild_indexes(self):
        """
        Example: reindexing entire DB — expensive operation; controlled by config flag.
        """
        if self.cfg.get("reindex", False):
            logger.info("Reindexing database...")
            cmd = ["bash", "-lc", f"psql \"{self.dsn}\" -c \"REINDEX DATABASE {self.cfg.get('database', 'postgres')};\""]
            run_cmd(cmd)

    def run_migrations(self):
        if not self.migration_cmd:
            logger.info("No migration command configured, skipping migrations.")
            return
        logger.info("Running migrations: %s", self.migration_cmd)
        run_cmd(["bash", "-lc", self.migration_cmd])
