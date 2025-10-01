# common/libraries/python/database/migrations.py
"""
Basic migration engine (SQLite/Postgres/MySQL).
"""

import os
from sqlalchemy import text
from .connection import db_connection
from .config import DatabaseConfig

class MigrationEngine:
    def __init__(self):
        self.migrations_dir = DatabaseConfig.MIGRATIONS_DIR
        os.makedirs(self.migrations_dir, exist_ok=True)

    def run_migrations(self):
        """Run all .sql migration files in order."""
        session = db_connection.get_session()
        try:
            for fname in sorted(os.listdir(self.migrations_dir)):
                if fname.endswith(".sql"):
                    with open(os.path.join(self.migrations_dir, fname), "r") as f:
                        sql = f.read()
                    session.execute(text(sql))
            session.commit()
        finally:
            session.close()

migration_engine = MigrationEngine()
