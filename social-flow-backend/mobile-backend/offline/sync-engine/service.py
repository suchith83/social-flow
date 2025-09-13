# Core service layer for sync orchestration
"""
Business logic for the sync engine.
Handles push (apply client ops), conflict resolution, and pull (delta) logic.
"""

from sqlalchemy.orm import Session
from .repository import SyncRepository
from .conflict import lww_strategy, merge_patch_strategy
from .models import PushBatchRequest, ChangeType, PullRequest, PullResponse
from typing import List, Dict, Any
from .changelog import to_change_record
from .config import get_config
import logging
from datetime import datetime, timezone

config = get_config()
logger = logging.getLogger("sync.service")


CONFLICT_STRATEGIES = {
    "lww": lww_strategy,
    "merge": merge_patch_strategy,
}


class SyncService:
    def __init__(self, db: Session, conflict_strategy: str = None):
        self.db = db
        self.conflict_strategy = conflict_strategy or config.DEFAULT_CONFLICT_STRATEGY
        self.strategy_fn = CONFLICT_STRATEGIES.get(self.conflict_strategy, lww_strategy)

    def _next_server_version(self) -> int:
        # simple approach: use current timestamp ms as server_version or a monotonic counter; choose timestamp for simplicity
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    def apply_push_batch(self, batch: PushBatchRequest) -> Dict[str, Any]:
        """
        Apply a client's ordered batch of operations atomically (best-effort).
        Returns a summary with applied seq range and conflicts list.
        """
        applied_changes = []
        conflicts = []
        # iterate ops in order
        for op in batch.ops:
            key = op.item.key
            incoming_payload = op.item.data
            client_version = op.item.client_version or 0
            server_item = SyncRepository.get_item(self.db, key)
            server_ver = server_item.server_version if server_item else 0
            target_server_version = self._next_server_version()

            # Decide based on op type
            if op.type == ChangeType.DELETE:
                # If server already has later version than client, we consider conflict
                if server_item and server_item.server_version > client_version:
                    # conflict: server has newer; decide via strategy
                    resolved_data, new_ver = self.strategy_fn(server_item, incoming_payload or {}, client_version)
                    # if resolved_data indicates deletion? For LWW, incoming delete may win if incoming_version >= server_version
                    if server_item.server_version <= client_version:
                        # accept delete
                        item = SyncRepository.mark_deleted(self.db, key, target_server_version)
                        log = SyncRepository.insert_changelog(self.db, key, ChangeType.DELETE, {}, target_server_version)
                        applied_changes.append(log)
                    else:
                        conflicts.append({"key": key, "reason": "server_newer_on_delete", "server_version": server_item.server_version})
                        # do not apply delete; still insert no-op log? skip
                else:
                    item = SyncRepository.mark_deleted(self.db, key, target_server_version)
                    log = SyncRepository.insert_changelog(self.db, key, ChangeType.DELETE, {}, target_server_version)
                    applied_changes.append(log)
            else:
                # CREATE or UPDATE
                # use conflict strategy to resolve
                resolved_data, new_ver = self.strategy_fn(server_item, incoming_payload, client_version)
                # new_ver may be computed; use target_server_version for server_version to keep monotonicity
                final_server_version = target_server_version
                # if resolved_data equals server and server had newer, consider conflict
                if server_item and server_item.server_version > client_version and resolved_data == (server_item.data or {}):
                    # server wins, treat as conflict
                    conflicts.append({"key": key, "reason": "server_wins", "server_version": server_item.server_version})
                    # still return current server state via changelog? No change needed.
                    continue
                # apply upsert
                item = SyncRepository.upsert_item(self.db, key, resolved_data, final_server_version, client_version, deleted=False)
                log = SyncRepository.insert_changelog(self.db, key, ChangeType.CREATE if op.type == ChangeType.CREATE else ChangeType.UPDATE, resolved_data, final_server_version)
                applied_changes.append(log)

        last_seq = SyncRepository.get_last_seq(self.db)
        return {
            "applied": len(applied_changes),
            "conflicts": conflicts,
            "last_seq": last_seq
        }

    def pull_changes(self, req: PullRequest) -> PullResponse:
        page_size = min(req.page_size or config.MAX_PULL_PAGE, config.MAX_PULL_PAGE)
        rows = SyncRepository.get_changes_since(self.db, req.since_seq, page_size)
        changes = [to_change_record(r) for r in rows]
        last_seq = SyncRepository.get_last_seq(self.db)
        return PullResponse(changes=changes, last_seq=last_seq)

    def compact_tombstones(self, older_than_timestamp: datetime):
        """
        Remove tombstones older than given timestamp from DB (physical cleanup).
        """
        removed = SyncRepository.delete_old_tombstones(self.db, older_than_timestamp)
        return {"removed": removed}
