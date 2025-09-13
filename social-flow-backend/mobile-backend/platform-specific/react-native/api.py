# FastAPI app exposing React Native-specific endpoints
"""
FastAPI app exposing React Native-specific endpoints.

Routes:
 - /register_device        POST  { raw_device_id, payload } -> register device & push token
 - /push/send              POST  { platform, min_rn, message } -> notify devices
 - /bundle/upload          POST  upload JS bundle (stores path)
 - /bundle/best_package    POST  { device_meta } -> full vs delta decision
 - /bundle/verify          POST  verify uploaded bundle checksum/signature
 - /analytics/ingest       POST  event or events
 - /deep_link/normalize    POST  normalize deep link
 - /metrics                GET aggregated metrics
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from typing import Dict, Any
import os
import time
import uuid
import hashlib
import asyncio

from .device_registry import RNDeviceRegistry
from .push import PushManager
from .bundle_diff import RNBundleDiffer
from .signature_verifier import SignatureVerifier
from .analytics import RNAnalyticsIngestor
from .deep_links import RNDeepLinkManager
from .metrics import RNMetrics
from .config import CONFIG
from .utils import ensure_dir, store_file, read_file, file_size, store, sha256_hex

app = FastAPI(title="React Native Platform-Specific Backend")

# instantiate components
device_registry = RNDeviceRegistry()
push_manager = PushManager(device_registry)
differ = RNBundleDiffer()
verifier = SignatureVerifier()
analytics = RNAnalyticsIngestor(device_registry)
deep_link_mgr = RNDeepLinkManager()
metrics = RNMetrics()

BUNDLE_DIR = CONFIG.bundle_storage_dir
ensure_dir(BUNDLE_DIR)

def _save_uploaded_bundle(uploaded: UploadFile) -> Dict[str, Any]:
    ext = os.path.splitext(uploaded.filename)[1] or ".bundle"
    name = f"{int(time.time())}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(BUNDLE_DIR, name)
    data = uploaded.file.read()
    store_file(path, data)
    return {"path": path, "size": len(data), "sha": sha256_hex(data), "filename": uploaded.filename}

# ---- endpoints ----
@app.post("/register_device")
async def register_device(body: Dict = Body(...)):
    raw = body.get("raw_device_id")
    payload = body.get("payload", {})
    if not raw:
        raise HTTPException(status_code=400, detail="missing raw_device_id")
    rec = device_registry.register(raw, payload)
    metrics.log("device_register", 1)
    return {"ok": True, "device": rec}

@app.post("/push/send")
async def push_send(body: Dict = Body(...)):
    platform = body.get("platform", "android")
    min_rn = body.get("min_rn")
    message = body.get("message", {})
    result = await push_manager.notify_query(platform=platform, message=message, min_rn=min_rn)
    metrics.log("push_send", 1)
    return {"ok": True, "result": result}

@app.post("/bundle/upload")
async def bundle_upload(file: UploadFile = File(...)):
    meta = _save_uploaded_bundle(file)
    index = store.get("rn_bundle_index") or {}
    ver = meta["sha"][:12]
    index[ver] = meta
    store.set("rn_bundle_index", index)
    metrics.log("bundle_upload", 1)
    return {"ok": True, "version": ver, "meta": meta}

@app.post("/bundle/best_package")
async def bundle_best_package(body: Dict = Body(...)):
    device_meta = body.get("device_meta", {})
    index = store.get("rn_bundle_index") or {}
    if not index:
        raise HTTPException(status_code=404, detail="no_bundles")
    decision = differ.best_package_for_client(device_meta, index)
    metrics.log("bundle_package_decision", 1)
    return decision

@app.post("/bundle/verify")
async def bundle_verify(file: UploadFile = File(...), expected_sha: str = Body(None)):
    tmp = _save_uploaded_bundle(file)
    path = tmp["path"]
    data = read_file(path) or b""
    ok_checksum, det = verifier.verify_checksum(data, expected_sha) if expected_sha else (True, {"note": "no_expected_sha_provided"})
    metrics.log("bundle_verify", 1 if ok_checksum else 0)
    return {"ok": ok_checksum, "details": det}

@app.post("/analytics/ingest")
async def analytics_ingest(body: Dict = Body(...)):
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

@app.get("/metrics")
async def get_metrics():
    return metrics.flush()

@app.get("/health")
async def health():
    return {"ok": True, "time": time.time()}
