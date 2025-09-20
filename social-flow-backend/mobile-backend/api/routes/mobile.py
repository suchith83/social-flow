from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time
import uuid

from mobile_backend.core.logger import get_logger
from mobile_backend.core.config import settings
from mobile_backend.notifications.push import PushSender

logger = get_logger("mobile.routes")
router = APIRouter()

# In-memory stores (simple & ephemeral)
_devices: Dict[str, Dict[str, Any]] = {}
_notifications: Dict[str, List[Dict[str, Any]]] = {}
_sync_store: Dict[str, Dict[str, Any]] = {}

push_sender = PushSender(provider=settings.PUSH_PROVIDER, creds=settings.PUSH_CREDENTIALS)


class DeviceRegister(BaseModel):
    device_id: str
    user_id: str
    platform: str = Field(..., description="ios | android")
    token: Optional[str] = Field(None, description="Push token / FCM token")


class SyncPayload(BaseModel):
    user_id: str
    last_synced_at: Optional[float]
    changes: Dict[str, Any] = {}


class PushPayload(BaseModel):
    user_id: str
    title: str
    body: str
    data: Dict[str, Any] = {}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/register_device")
def register_device(payload: DeviceRegister):
    # upsert device
    logger.info("Register device %s for user %s", payload.device_id, payload.user_id)
    _devices[payload.device_id] = {
        "device_id": payload.device_id,
        "user_id": payload.user_id,
        "platform": payload.platform,
        "token": payload.token,
        "registered_at": time.time(),
    }
    return {"status": "registered", "device_id": payload.device_id}


@router.post("/sync")
def sync(payload: SyncPayload):
    # lightweight sync endpoint: accept client-side changes and return server-side updates
    uid = payload.user_id
    logger.info("Sync for user=%s last_synced=%s", uid, payload.last_synced_at)
    _sync_store.setdefault(uid, {})
    # persist changes in-memory (in production enqueue to DB / event store)
    _sync_store[uid].update(payload.changes or {})
    # return simple server state (could be delta)
    server_state = {"server_time": time.time(), "data": _sync_store[uid]}
    return {"status": "ok", "state": server_state}


@router.post("/push")
def push(payload: PushPayload):
    # find devices for user and send push
    devices = [d for d in _devices.values() if d.get("user_id") == payload.user_id]
    if not devices:
        raise HTTPException(status_code=404, detail="no devices for user")
    sent = []
    for d in devices:
        try:
            push_sender.send(token=d.get("token"), title=payload.title, body=payload.body, data=payload.data)
            # record notification in memory for retrieval
            _notifications.setdefault(payload.user_id, []).append({
                "id": f"nt_{uuid.uuid4().hex[:10]}",
                "title": payload.title,
                "body": payload.body,
                "data": payload.data,
                "device_id": d.get("device_id"),
                "sent_at": time.time()
            })
            sent.append(d.get("device_id"))
        except Exception:
            logger.exception("Failed to send push to device=%s", d.get("device_id"))
    return {"status": "sent", "devices": sent}


@router.get("/notifications/{user_id}")
def list_notifications(user_id: str, limit: int = 50):
    items = list(reversed(_notifications.get(user_id, [])))[:limit]
    return {"user_id": user_id, "count": len(items), "notifications": items}
