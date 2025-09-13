# CLI for cache operations
"""
Small CLI helpers for admin tasks: prefill from manifest, show stats, cleanup.
Useful during development.
"""

import argparse
import json
import os
from .database import SessionLocal, init_db
from .service import ContentCacheService
from .config import get_config
from .storage import DiskStore

def show_stats():
    init_db()
    db = SessionLocal()
    svc = ContentCacheService(db)
    disk = DiskStore()
    print("Disk total size:", disk.total_size())
    db.close()

def prefill(manifest_path: str):
    """
    Manifest is a JSON array of objects: [{"key": "...", "url": "..."}, ...]
    """
    init_db()
    db = SessionLocal()
    svc = ContentCacheService(db)
    with open(manifest_path, "r") as f:
        entries = json.load(f)
    keys = {e["key"]: {"url": e["url"]} for e in entries}
    svc.prefetch_keys(list(keys.keys()), origin_map=keys)
    db.close()

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("stats")
    pfill = sub.add_parser("prefill")
    pfill.add_argument("manifest")
    args = parser.parse_args()
    if args.cmd == "stats":
        show_stats()
    elif args.cmd == "prefill":
        prefill(args.manifest)

if __name__ == "__main__":
    main()
