# Conflict resolution logic for sync
"""
Conflict resolution strategies for the sync engine.
Export functions that take (server_item, incoming_item, op) and return resolved data + server_version bump.
"""

from typing import Tuple, Optional, Dict, Any
from .models import SyncItem
import logging

logger = logging.getLogger("sync.conflict")


def lww_strategy(server_item: Optional[SyncItem], incoming: Dict[str, Any], incoming_version: int) -> Tuple[Dict[str, Any], int]:
    """
    Last-write-wins: compare incoming_version vs server.server_version; higher wins.
    incoming_version is client's lamport/clock. We choose the incoming if incoming_version >= server.server_version.
    Return (resolved_data, new_server_version).
    """
    if server_item is None:
        # new item; assign version = incoming_version + 1
        new_ver = incoming_version + 1
        return incoming, new_ver
    if incoming_version >= server_item.server_version:
        new_ver = incoming_version + 1
        return incoming, new_ver
    # keep server
    return server_item.data or {}, server_item.server_version


def merge_patch_strategy(server_item: Optional[SyncItem], incoming: Dict[str, Any], incoming_version: int) -> Tuple[Dict[str, Any], int]:
    """
    Merge strategy that treats 'incoming' as a partial patch and shallow-merges into server data.
    Bumps version on merge.
    """
    if server_item is None:
        new_ver = incoming_version + 1
        return incoming, new_ver
    base = dict(server_item.data or {})
    base.update(incoming)  # shallow merge
    new_ver = max(server_item.server_version, incoming_version) + 1
    return base, new_ver
