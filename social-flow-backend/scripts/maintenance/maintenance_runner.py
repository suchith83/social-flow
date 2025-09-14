# scripts/maintenance/maintenance_runner.py
import logging
import sys
import argparse
from typing import Dict, Any

from .config_loader import ConfigLoader
from .lockfile import file_lock
from .notifier import Notifier
from .backup_manager import BackupManager
from .db_maintenance import DBMaintenance
from .log_rotator import LogRotator
from .cleanup import Cleaner
from .scheduler import Scheduler

logger = logging.getLogger("maintenance")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_LOCK = "/var/lock/socialflow-maintenance.lock"

def run_all(config_path: str, dry_run: bool = False):
    cfg = ConfigLoader(config_path).load()
    notifier = Notifier(cfg)
    lock_path = cfg.get("maintenance", {}).get("lockfile", DEFAULT_LOCK)
    try:
        with file_lock(lock_path, timeout=300):
            notifier.notify("🔧 Starting maintenance run")
            # Backup
            backup_cfg = cfg.get("maintenance", {}).get("backup", {})
            if backup_cfg.get("enabled", False):
                bm = BackupManager(cfg)
                logger.info("Starting backups")
                # Sample: directories from config
                dirs = backup_cfg.get("directories", [])
                for d in dirs:
                    bm.backup_directories("files", [d], upload=backup_cfg.get("upload", True))
                # Optionally database
                if backup_cfg.get("postgres_dsn"):
                    bm.backup_postgres("postgres", backup_cfg.get("postgres_dsn"), upload=backup_cfg.get("upload", True))
                bm.prune_local()
            else:
                logger.info("Backups disabled in config")

            # DB maintenance
            dbm = DBMaintenance(cfg)
            dbm.run_migrations()
            dbm.run_vacuum()
            dbm.rebuild_indexes()

            # Rotate logs
            lr = LogRotator(cfg)
            lr.rotate_all()
            lr.prune()

            # Cleanup
            cleaner = Cleaner(cfg)
            if dry_run:
                cleaner.dry_run = True
            cleaner.run()

            notifier.notify("✅ Maintenance run completed successfully")
    except Exception as e:
        logger.exception("Maintenance failed: %s", e)
        notifier.notify(f"❌ Maintenance failed: {e}")
        raise

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run maintenance tasks")
    parser.add_argument("--config", default="maintenance.yaml", help="Path to maintenance config")
    parser.add_argument("--dry-run", action="store_true", help="Do not delete or modify resources")
    parser.add_argument("--once", action="store_true", help="Run once and exit (default behavior)")
    args = parser.parse_args(argv)

    try:
        run_all(args.config, dry_run=args.dry_run)
    except Exception as e:
        logging.error("Run failed: %s", e)
        sys.exit(2)

if __name__ == "__main__":
    main()
