# CLI for sync-engine operations
"""
CLI helpers for admin: show last seq, run tombstone cleanup, compact DB.
"""

import argparse
from .database import SessionLocal, init_db
from .tasks import tombstone_cleanup_task
from .service import SyncService
from datetime import datetime, timezone, timedelta

def show_last_seq():
    init_db()
    db = SessionLocal()
    from .repository import SyncRepository
    last = SyncRepository.get_last_seq(db)
    print("Last global seq:", last)
    db.close()

def cleanup_tombstones(days: int = 7):
    init_db()
    db = SessionLocal()
    svc = SyncService(db)
    cutoff = datetime.now(timezone=timezone.utc) - timedelta(days=days)
    res = svc.compact_tombstones(cutoff)
    print("Tombstone cleanup:", res)
    db.close()

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("last-seq")
    p = sub.add_parser("cleanup")
    p.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    if args.cmd == "last-seq":
        show_last_seq()
    elif args.cmd == "cleanup":
        cleanup_tombstones(args.days)

if __name__ == "__main__":
    main()
