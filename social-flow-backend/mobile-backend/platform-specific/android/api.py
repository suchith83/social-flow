# FastAPI app exposing Android-specific endpoints
"""
FastAPI app exposing Android-specific endpoints.

Routes:
 - /register_device         POST  { raw_device_id, payload } -> register device & push token
 - /push/send               POST  { min_android, message } -> notify devices
 - /apk/upload              POST  upload an APK file, store metadata
 - /apk/best_package        POST  { device_meta } -> returns full APK path or diff path decision
 - /apk/verify              POST  verify uploaded APK signature
 - /analytics/ingest        POST  analytics events
 - /deep_link/normalize     POST  normalize deep link

This module wires together the other helpers above. In production, mount this into your ASGI server.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any
import os
import time
import hashlib
import uuid
import asyncio

from .device_registry import DeviceRegistry
from .fcm import FCMManager
from .apk_diff import ApkDiffer
from .signature_verifier import SignatureVerifier
from .analytics import AnalyticsIngestor
from .deep_links import DeepLinkManager
from .metrics import AndroidMetrics
from .config import CONFIG
from .utils import ensure_dir, store_file, read_file, file_size, store

app = FastAPI(title="Android Platform-Specific Backend")

# instantiate components (could be injected)
device_registry = DeviceRegistry()
fcm = FCMManager(device_registry)
differ = ApkDiffer()
verifier = SignatureVerifier()
analytics = AnalyticsIngestor(device_registry)
deep_link_mgr = DeepLinkManager()
metrics = AndroidMetrics()

APK_DIR = CONFIG.apk_storage_dir
ensure_dir(APK_DIR)

# ---- helper ----
def _save_uploaded_apk(uploaded: UploadFile) -> Dict[str, Any]:
    """
    Save uploaded APK into storage dir and return metadata (path, sha)
    """
    ext = os.path.splitext(uploaded.filename)[1] or ".apk"
    name = f"{int(time.time())}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(APK_DIR, name)
    data = uploaded.file.read()
    store_file(path, data)
    return {"path": path, "size": len(data), "sha": hashlib.sha256(data).hexdigest(), "filename": uploaded.filename}

# ---- endpoints ----
@app.post("/register_device")
async def register_device(body: Dict = Body(...)):
    """
    body: {
       "raw_device_id": "...",
       "payload": {...}
    }
    """
    raw = body.get("raw_device_id")
    payload = body.get("payload", {})
    if not raw:
        raise HTTPException(status_code=400, detail="missing raw_device_id")
    rec = device_registry.register(raw, payload)
    metrics.log("device_register", 1)
    return {"ok": True, "device": rec}

@app.post("/push/send")
async def push_send(body: Dict = Body(...)):
    """
    Send push to devices matching min_android (optional).
    body: {"min_android": 21, "message": {"title":"Hi","body":"Hello"}}
    """
    min_android = body.get("min_android", 0)
    message = body.get("message", {})
    # Use FCMManager to gather tokens and send
    result = await fcm.notify_devices_by_query(min_android=min_android, message=message)
    metrics.log("push_send", 1)
    return {"ok": True, "result": result}

@app.post("/apk/upload")
async def apk_upload(file: UploadFile = File(...)):
    """
    Upload APK. Stores it and returns metadata.
    """
    meta = _save_uploaded_apk(file)
    # store available versions in in-memory index
    index = store.get("apk_index") or {}
    # name the version id by sha prefix
    ver = meta["sha"][:12]
    index[ver] = meta
    store.set("apk_index", index)
    metrics.log("apk_upload", 1)
    return {"ok": True, "version": ver, "meta": meta}

@app.post("/apk/best_package")
async def apk_best_package(body: Dict = Body(...)):
    """
    Given a client device meta, decide whether to serve full APK or diff.
    body: {"device_meta": {"installed_sha": "...", ...}}
    """
    device_meta = body.get("device_meta", {})
    index = store.get("apk_index") or {}
    if not index:
        raise HTTPException(status_code=404, detail="no_apks_available")
    decision = differ.best_package_for_client(device_meta, index)
    metrics.log("apk_package_decision", 1)
    return decision

@app.post("/apk/verify")
async def apk_verify(file: UploadFile = File(...)):
    """
    Verify signature of an uploaded APK (temp). Returns verification result.
    """
    tmp_meta = _save_uploaded_apk(file)
    path = tmp_meta["path"]
    ok, details = verifier.verify_signature(path)
    metrics.log("apk_verify", 1 if ok else 0)
    return {"ok": ok, "details": details}

@app.post("/analytics/ingest")
async def analytics_ingest(body: Dict = Body(...)):
    """
    Ingest Android analytics event (single event or batch)
    body: {"event": {...}}  or {"events": [{...}, ...]}
    """
    if "event" in body:
        ev = body["event"]
        res = await analytics.ingest(ev)
        metrics.log("analytics_event", 1 if res.get("ok") else 0)
        return res
    elif "events" in body:
        out = []
        for ev in body["events"]:
            r = await analytics.ingest(ev)
            out.append(r)
        metrics.log("analytics_batch", len(body["events"]))
        return {"results": out}
    else:
        raise HTTPException(status_code=400, detail="missing event(s)")

@app.post("/deep_link/normalize")
async def deep_link_normalize(body: Dict = Body(...)):
    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="missing url")
    normalized = deep_link_mgr.normalize(url)
    if not normalized:
        raise HTTPException(status_code=400, detail="invalid_deep_link")
    metrics.log("deep_link_normalize", 1)
    return {"ok": True, "normalized": normalized}

# simple health & metrics endpoints
@app.get("/health")
async def health():
    return {"ok": True, "time": time.time()}

@app.get("/metrics")
async def get_metrics():
    return metrics.flush()
