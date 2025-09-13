"""Handles schema migrations (Alembic + CockroachDB)."""
"""
migration_manager.py
---------------------
Handles schema migrations for CockroachDB using Alembic.
Ensures compatibility with distributed SQL.
"""

import subprocess
import logging

logger = logging.getLogger("MigrationManager")
logger.setLevel(logging.INFO)


class MigrationManager:
    """Wrapper for Alembic migrations tailored for CockroachDB."""

    @staticmethod
    def upgrade(revision="head"):
        """Run Alembic upgrade."""
        logger.info(f"Running Alembic upgrade to {revision}")
        subprocess.run(["alembic", "upgrade", revision], check=True)

    @staticmethod
    def downgrade(revision="-1"):
        """Rollback last migration."""
        logger.info(f"Downgrading Alembic to {revision}")
        subprocess.run(["alembic", "downgrade", revision], check=True)

    @staticmethod
    def generate(message="auto migration"):
        """Generate new migration script."""
        logger.info("Generating new migration script")
        subprocess.run(["alembic", "revision", "--autogenerate", "-m", message], check=True)
