# FastAPI app exposing Flutter-specific endpoints
"""
FastAPI app for Flutter-specific backend endpoints.

Routes:
 - /register_device       POST { raw_device_id, payload } -> register Flutter device
 - /push/send             POST { platform, min_flutter, message } -> notify devices
 - /bundle/upload         POST upload bundle (AAB/IPA/flutter bundle)
 - /bundle/best_package   POST { device_meta } -> return full bundle or diff path
 - /bundle/verify         POST verify uploaded bundle signature
 - /analytics/ingest      POST event or batch
 - /deep_link/normalize   POST normalize deep link
 - /metrics               GET aggregated metrics
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any
import os
import time
import uuid
import hashlib
import asyncio

from .device_registry import FlutterDeviceRegistry
from .push import PushManager
from .bundle_diff import FlutterBundleDiffer
from .signature_verifier import FlutterSignatureVerifier
from .analytics import FlutterAnalyticsIngestor
from .deep_links import FlutterDeepLinkManager
from .metrics import FlutterMetrics
from .config import CONFIG
from .utils import ensure_dir, store_file, read_file, file_size, store

app = FastAPI(title="Flutter Platform-Specific Backend")

# instantiate components
device_registry = FlutterDeviceRegistry()
push_manager = PushManager(device_registry)
differ = FlutterBundleDiffer()
verifier = FlutterSignatureVerifier()
analytics = FlutterAnalyticsIngestor(device_registry)
deep_link_mgr = FlutterDeepLinkManager()
metrics = FlutterMetrics()

BUNDLE_DIR = CONFIG.bundle_storage_dir
ensure_dir(BUNDLE_DIR)

def _save_uploaded_bundle(uploaded: UploadFile) -> Dict[str, Any]:
    ext = os.path.splitext(uploaded.filename)[1] or ".bundle"
    name = f"{int(time.time())}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(BUNDLE_DIR, name)
    data = uploaded.file.read()
    store_file(path, data)
    return {"path": path, "size": len(data), "sha": hashlib.sha256(data).hexdigest(), "filename": uploaded.filename}

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
    platform = body.get("platform")
    min_flutter = body.get("min_flutter")
    message = body.get("message", {})
    res = await push_manager.notify_query(platform=platform, message=message, min_flutter=min_flutter)
    metrics.log("push_send", 1)
    return {"ok": True, "result": res}

@app.post("/bundle/upload")
async def bundle_upload(file: UploadFile = File(...)):
    meta = _save_uploaded_bundle(file)
    index = store.get("bundle_index") or {}
    ver = meta["sha"][:12]
    index[ver] = meta
    store.set("bundle_index", index)
    metrics.log("bundle_upload", 1)
    return {"ok": True, "version": ver, "meta": meta}

@app.post("/bundle/best_package")
async def bundle_best_package(body: Dict = Body(...)):
    device_meta = body.get("device_meta", {})
    index = store.get("bundle_index") or {}
    if not index:
        raise HTTPException(status_code=404, detail="no_bundles")
    decision = differ.best_package_for_client(device_meta, index)
    metrics.log("bundle_package_decision", 1)
    return decision

@app.post("/bundle/verify")
async def bundle_verify(file: UploadFile = File(...)):
    tmp = _save_uploaded_bundle(file)
    path = tmp["path"]
    # try android apk verification first, then ipa fallback
    ok_a, det_a = verifier.verify_android_apk(path)
    ok_i, det_i = verifier.verify_ios_ipa(path)
    ok = ok_a or ok_i
    details = {"android": det_a, "ios": det_i}
    metrics.log("bundle_verify", 1 if ok else 0)
    return {"ok": ok, "details": details}

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
