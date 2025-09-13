"""Automated backup and restore using InfluxDB native tools."""
"""
backup_manager.py
-----------------
Automates InfluxDB backups & restores using CLI integration.
"""

import subprocess
import logging

logger = logging.getLogger("InfluxDBBackupManager")
logger.setLevel(logging.INFO)


class BackupManager:
    def __init__(self, backup_dir="/backups/influxdb"):
        self.backup_dir = backup_dir

    def run_backup(self):
        """Perform backup using influx CLI."""
        cmd = ["influx", "backup", self.backup_dir]
        subprocess.run(cmd, check=True)
        logger.info(f"✅ Backup completed at {self.backup_dir}")

    def run_restore(self, backup_dir=None):
        """Restore from backup using influx CLI."""
        restore_dir = backup_dir or self.backup_dir
        cmd = ["influx", "restore", "--full", restore_dir]
        subprocess.run(cmd, check=True)
        logger.info(f"🔄 Restore completed from {restore_dir}")
