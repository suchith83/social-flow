# FastAPI app exposing iOS-specific endpoints
"""
FastAPI app exposing iOS-specific endpoints.

Routes:
 - /register_device         POST  { raw_device_id, payload } -> register device & APNs token
 - /push/send               POST  { min_ios, message } -> notify devices
 - /ipa/upload              POST  upload an IPA file, store metadata
 - /ipa/best_package        POST  { device_meta } -> returns full IPA path or diff path decision
 - /ipa/verify              POST  verify uploaded IPA signature
 - /analytics/ingest        POST  analytics events
 - /universal_link/normalize POST normalize deep link
 - /metrics                 GET aggregated metrics
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any
import os
import time
import hashlib
import uuid
import asyncio

from .device_registry import iOSDeviceRegistry
from .apns import APNSManager
from .ipa_diff import IpaDiffer
from .signature_verifier import IPASignatureVerifier
from .analytics import AnalyticsIngestor
from .deep_links import UniversalLinkManager
from .metrics import iOSMetrics
from .config import CONFIG
from .utils import ensure_dir, store_file, read_file, file_size, store

app = FastAPI(title="iOS Platform-Specific Backend")

# instantiate components
device_registry = iOSDeviceRegistry()
apns = APNSManager(device_registry)
differ = IpaDiffer()
verifier = IPASignatureVerifier()
analytics = AnalyticsIngestor(device_registry)
link_mgr = UniversalLinkManager()
metrics = iOSMetrics()

IPA_DIR = CONFIG.ipa_storage_dir
ensure_dir(IPA_DIR)

def _save_uploaded_ipa(uploaded: UploadFile) -> Dict[str, Any]:
    ext = os.path.splitext(uploaded.filename)[1] or ".ipa"
    name = f"{int(time.time())}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(IPA_DIR, name)
    data = uploaded.file.read()
    store_file(path, data)
    return {"path": path, "size": len(data), "sha": hashlib.sha256(data).hexdigest(), "filename": uploaded.filename}

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
    """
    body: {"min_ios": "14.0", "message": {"aps": {"alert": {"title":"Hi","body":"Yo"}}}}
    """
    min_ios = body.get("min_ios")
    message = body.get("message", {})
    result = await apns.notify_devices_by_query(min_ios=min_ios, message=message)
    metrics.log("push_send", 1)
    return {"ok": True, "result": result}

@app.post("/ipa/upload")
async def ipa_upload(file: UploadFile = File(...)):
    meta = _save_uploaded_ipa(file)
    index = store.get("ipa_index") or {}
    ver = meta["sha"][:12]
    index[ver] = meta
    store.set("ipa_index", index)
    metrics.log("ipa_upload", 1)
    return {"ok": True, "version": ver, "meta": meta}

@app.post("/ipa/best_package")
async def ipa_best_package(body: Dict = Body(...)):
    device_meta = body.get("device_meta", {})
    index = store.get("ipa_index") or {}
    if not index:
        raise HTTPException(status_code=404, detail="no_ipas_available")
    decision = differ.best_package_for_client(device_meta, index)
    metrics.log("ipa_package_decision", 1)
    return decision

@app.post("/ipa/verify")
async def ipa_verify(file: UploadFile = File(...)):
    tmp_meta = _save_uploaded_ipa(file)
    path = tmp_meta["path"]
    ok, details = verifier.verify_signature(path)
    metrics.log("ipa_verify", 1 if ok else 0)
    return {"ok": ok, "details": details}

@app.post("/analytics/ingest")
async def analytics_ingest(body: Dict = Body(...)):
    """
    body: {"event": {...}} or {"events":[{...}, ...]}
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

@app.post("/universal_link/normalize")
async def universal_link_normalize(body: Dict = Body(...)):
    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="missing url")
    normalized = link_mgr.normalize(url)
    if not normalized:
        raise HTTPException(status_code=400, detail="invalid_universal_link")
    metrics.log("universal_link_normalize", 1)
    return {"ok": True, "normalized": normalized}

@app.get("/metrics")
async def get_metrics():
    return metrics.flush()

@app.get("/health")
async def health():
    return {"ok": True, "time": time.time()}
