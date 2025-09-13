"""Safe migrations for schema/indices (mongock-style or Python wrapper)."""
"""
migration_manager.py
--------------------
Migration framework for MongoDB collections. Unlike SQL, MongoDB schema
migrations typically are code migrations: alter documents, create indexes,
migrate data, or backfill fields.

This module provides:
- A simple migration runner that loads migration scripts from migrations/
- Idempotency guarantees (each migration records execution in a `migrations` collection)
- Safe runner with dry-run option and batch-size controlled updates
- Example migration script format included in comments
"""

import importlib.util
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import List
from .connection import MongoConnectionManager

logger = logging.getLogger("MongoMigrations")
logger.setLevel(logging.INFO)

MIGRATIONS_DIR = Path("data/migrations/mongodb")


def _ensure_migrations_collection(db):
    if "migrations" not in db.list_collection_names():
        db.create_collection("migrations")


def _applied_migrations(db) -> List[str]:
    _ensure_migrations_collection(db)
    return [m["name"] for m in db["migrations"].find({}, {"name": 1, "_id": 0})]


def _record_migration(db, name: str):
    db["migrations"].insert_one({"name": name, "applied_at": datetime.utcnow()})


def _load_migration_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pending(dry_run: bool = False, batch_size: int = 1000):
    """
    Iterate over migration files in `data/migrations/mongodb` sorted by filename,
    run those not yet applied. Each migration module should implement `up(db, batch_size)`
    and optionally `down(db)`.

    Migration file example (save as e.g. `001_add_user_display_name.py`):
    --------------------------------------------------------------
    def up(db, batch_size=1000):
        # add display_name field for users who don't have it yet in batches
        cursor = db.users.find({"display_name": {"$exists": False}}, projection=["_id"])
        batch = []
        for doc in cursor:
            batch.append(doc["_id"])
            if len(batch) >= batch_size:
                db.users.update_many({"_id": {"$in": batch}}, {"$set": {"display_name": ""}})
                batch = []
        if batch:
            db.users.update_many({"_id": {"$in": batch}}, {"$set": {"display_name": ""}})
    --------------------------------------------------------------
    """
    conn = MongoConnectionManager()
    db = conn.get_database()
    applied = set(_applied_migrations(db))

    if not MIGRATIONS_DIR.exists():
        logger.warning("Migrations directory does not exist")
        return

    files = sorted([p for p in MIGRATIONS_DIR.iterdir() if p.suffix == ".py"])
    for file in files:
        name = file.name
        if name in applied:
            logger.info(f"Skipping already applied migration: {name}")
            continue

        logger.info(f"Applying migration: {name}")
        module = _load_migration_module(file)
        if dry_run:
            logger.info(f"Dry run mode: would execute migration {name}")
            continue

        try:
            if hasattr(module, "up"):
                module.up(conn.get_database(), batch_size=batch_size)
                _record_migration(conn.get_database(), name)
                logger.info(f"Migration {name} applied successfully")
            else:
                logger.warning(f"Migration {name} does not implement up(db)")
        except Exception as e:
            logger.exception(f"Failed to apply migration {name}: {e}")
            raise
