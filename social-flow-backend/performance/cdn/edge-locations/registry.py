# Manages registration and metadata of edge locations
# performance/cdn/edge-locations/registry.py
"""
Edge registry: authoritative store (in-memory + pluggable persistence adapter)
for edge nodes metadata and state. Provides thread-safe CRUD and search.

Design notes:
- Default uses an in-memory dict with asyncio.Lock for concurrency.
- Adapter pattern allows swapping in Redis / etcd / SQL for persistence.
- Includes basic TTL eviction and health status.
"""

from typing import Dict, Optional, List, Callable, Any
from .utils import EdgeNode, logger, is_valid_ip
import asyncio
import time
import uuid

_DEFAULT_TTL_SECONDS = 300

class InMemoryAdapter:
    """Simple persistence adapter: stores node dicts in memory."""
    def __init__(self):
        self.store: Dict[str, Dict] = {}

    async def get(self, node_id: str) -> Optional[Dict]:
        return self.store.get(node_id)

    async def upsert(self, node_id: str, data: Dict):
        self.store[node_id] = data

    async def delete(self, node_id: str):
        if node_id in self.store:
            del self.store[node_id]

    async def list_all(self) -> List[Dict]:
        return list(self.store.values())

class EdgeRegistry:
    def __init__(self, adapter: Optional[Any] = None, ttl_seconds: int = _DEFAULT_TTL_SECONDS):
        self.adapter = adapter or InMemoryAdapter()
        self.ttl_seconds = ttl_seconds
        self.lock = asyncio.Lock()

    async def register(self, node: EdgeNode) -> str:
        """Register or update an edge node. Returns node id."""
        if not is_valid_ip(node.ip):
            raise ValueError("Invalid IP address")
        node_id = node.id or str(uuid.uuid4())
        node_dict = node.to_dict()
        node_dict["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        async with self.lock:
            await self.adapter.upsert(node_id, node_dict)
        logger.info(f"Registered node {node_id} ({node.hostname})")
        return node_id

    async def get(self, node_id: str) -> Optional[EdgeNode]:
        async with self.lock:
            rec = await self.adapter.get(node_id)
        if not rec:
            return None
        return EdgeNode(**rec)

    async def update_load(self, node_id: str, load_rps: float):
        async with self.lock:
            rec = await self.adapter.get(node_id)
            if not rec:
                raise KeyError("Node not found")
            rec["current_load_rps"] = load_rps
            rec["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            await self.adapter.upsert(node_id, rec)

    async def remove(self, node_id: str):
        async with self.lock:
            await self.adapter.delete(node_id)
            logger.info(f"Removed node {node_id}")

    async def list_nodes(self, region: Optional[str] = None, healthy_only: bool = True) -> List[EdgeNode]:
        async with self.lock:
            recs = await self.adapter.list_all()
        nodes = [EdgeNode(**r) for r in recs]
        if region:
            nodes = [n for n in nodes if n.region == region]
        if healthy_only:
            nodes = [n for n in nodes if n.healthy]
        return nodes

    async def evict_stale(self):
        """Evict nodes not seen within TTL."""
        now_ts = time.time()
        async with self.lock:
            recs = await self.adapter.list_all()
            for r in recs:
                last_seen = r.get("last_seen")
                if not last_seen:
                    continue
                # last_seen stored as ISO Z: parse simply
                try:
                    last_epoch = time.mktime(time.strptime(last_seen, "%Y-%m-%dT%H:%M:%SZ"))
                except Exception:
                    continue
                if now_ts - last_epoch > self.ttl_seconds:
                    await self.adapter.delete(r["id"])
                    logger.info(f"Evicted stale node {r['id']}")
