# Track changes for sync operations
"""
Helpers to produce ChangeRecord objects for API responses.
"""

from .models import ChangeLog, ChangeType
from .models import ChangeRecord
from datetime import datetime


def to_change_record(row: ChangeLog) -> ChangeRecord:
    return ChangeRecord(
        seq=row.seq,
        key=row.key,
        change_type=ChangeType(row.change_type),
        payload=row.payload,
        server_version=row.server_version,
        created_at=row.created_at
    )
