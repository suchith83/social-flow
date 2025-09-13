"""Replica set health, primary/secondary checks, replication lag metrics."""
"""
replication_monitor.py
----------------------
Utilities to inspect replica set health, find primary/secondaries, and compute
replication lag statistics. Intentionally lightweight — integrate into your
monitoring system (Prometheus exporters, alerting) as needed.

Provides:
  - get_replica_set_status()
  - get_primary_info()
  - compute_replication_lag_seconds()
"""

from .connection import MongoConnectionManager
import logging
from datetime import datetime, timezone

logger = logging.getLogger("MongoReplication")
logger.setLevel(logging.INFO)


def get_replica_set_status():
    conn = MongoConnectionManager()
    admin_db = conn.get_sync_client().admin
    status = admin_db.command("replSetGetStatus")
    return status


def get_primary_info():
    status = get_replica_set_status()
    for member in status.get("members", []):
        if member.get("stateStr") == "PRIMARY":
            return {
                "name": member.get("name"),
                "optimeDate": member.get("optimeDate"),
                "health": member.get("health"),
                "state": member.get("stateStr")
            }
    return None


def compute_replication_lag_seconds():
    """
    Compute lag as difference between primary optimeDate and each secondary optimeDate.
    Returns list of dicts: {member, lag_seconds}
    """
    status = get_replica_set_status()
    primary_optime = None
    for m in status.get("members", []):
        if m.get("stateStr") == "PRIMARY":
            primary_optime = m.get("optimeDate")
            break

    if not primary_optime:
        raise RuntimeError("Primary not found in replica set status")

    results = []
    for m in status.get("members", []):
        if m.get("optimeDate") is None:
            lag = None
        else:
            lag = (primary_optime - m.get("optimeDate")).total_seconds()
        results.append({
            "name": m.get("name"),
            "state": m.get("stateStr"),
            "lag_seconds": lag
        })
    return results
