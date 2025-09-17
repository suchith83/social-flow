"""
FastAPI endpoints for thumbnail operations.

Endpoints:
- POST /thumbnails/generate -> request generation (sync or async)
- GET  /thumbnails/{video_id} -> list thumbnails & metadata (presigned URLs)
- GET  /thumbnails/{video_id}/{thumb} -> presigned or direct retrieval (via storage presign)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import os
import uuid

from .generator import ThumbnailGenerator
from .models import ThumbnailSpec, ThumbnailsBatchResult, ThumbnailResult
from .storage_uploader import ThumbnailStorage
from .jobs import enqueue, start_worker_in_thread, send_callback
from .config import config
from .utils import ensure_dir, logger

app = FastAPI(title="Video Thumbnails API")
gen = ThumbnailGenerator()
store = ThumbnailStorage()

# Ensure worker is started in-process for demo convenience
start_worker_in_thread()


class GenerateRequest(BaseModel):
    video_id: str
    video_path: str  # local path to staged video (server-side); for direct S3 files, download or stream first
    timestamps: Optional[List[float]] = None
    count: Optional[int] = 5
    sizes: Optional[List[str]] = None  # ["320x180","640x360"]
    formats: Optional[List[str]] = ["jpeg"]
    async_job: Optional[bool] = True


def _make_specs(sizes: Optional[List[str]], formats: List[str]) -> List[ThumbnailSpec]:
    specs = []
    sizes = sizes or config.DEFAULT_SIZES.split(",")
    for s in sizes:
        s = s.strip()
        if not s:
            continue
        w,h = s.split("x")
        for fmt in formats:
            specs.append(ThumbnailSpec(width=int(w), height=int(h), format=fmt))
    return specs


def _upload_and_build_result(video_id: str, local_paths: List[str]) -> ThumbnailsBatchResult:
    results = []
    for p in local_paths:
        # derive key and upload
        key = os.path.join(video_id, os.path.basename(p))
        url = store.upload_file(p, key)
        # try to parse width/height from filename pattern; otherwise leave None
        # we rely on generator providing phash if enabled
        thumb_id = os.path.splitext(os.path.basename(p))[0].split("_")[0]
        # fake parse size e.g. id_320x180.jpeg
        parts = os.path.basename(p).split("_")
        size_part = None
        fmt = os.path.splitext(p)[1].lstrip(".")
        if len(parts) >= 2:
            size_part = parts[1].split(".")[0]
        res = ThumbnailResult(video_id=video_id, thumbnail_id=thumb_id, size=size_part or "", format=fmt, url=url)
        results.append(res)
    sprite_url = None
    # optionally build sprite if allowed
    if config.ALLOW_SPRITE and len(local_paths) > 1:
        try:
            from .sprite import generate_contact_sheet
            sprite_path = os.path.join(config.OUTPUT_DIR, video_id, "sprite.jpg")
            meta = generate_contact_sheet(local_paths, sprite_path, cols=4)
            sprite_key = os.path.join(video_id, os.path.basename(sprite_path))
            sprite_url = store.upload_file(sprite_path, sprite_key)
        except Exception as e:
            logger.warning("Sprite generation failed: %s", e)
            sprite_url = None
    return ThumbnailsBatchResult(video_id=video_id, thumbnails=results, sprite_url=sprite_url, metadata={})


def _job_generate_and_upload(video_id: str, video_path: str, timestamps, specs):
    try:
        local_results = []
        if timestamps:
            res_objs = gen.extract_at_timestamps(video_path, video_id, timestamps, specs)
        else:
            res_objs = gen.extract_evenly_spaced(video_path, video_id, count=len(specs)//len(set([s.format for s in specs])) or 5, specs=specs)
        local_paths = [r.url for r in res_objs]  # r.url currently local path
        batch = _upload_and_build_result(video_id, local_paths)
        payload = {"video_id": video_id, "status": "completed", "result": batch.dict()}
        send_callback(payload)
    except Exception as e:
        logger.exception("Thumbnail job failed for %s: %s", video_id, e)
        send_callback({"video_id": video_id, "status": "failed", "error": str(e)})


@app.post("/thumbnails/generate", response_model=ThumbnailsBatchResult)
def generate_thumbnails(req: GenerateRequest, background_tasks: BackgroundTasks):
    specs = _make_specs(req.sizes, req.formats)
    # cap number of thumbnails
    if len(specs) > config.MAX_THUMBNAILS:
        specs = specs[:config.MAX_THUMBNAILS]

    if req.async_job:
        # enqueue background work
        enqueue(_job_generate_and_upload, req.video_id, req.video_path, req.timestamps, specs)
        # respond with pending metadata (empty results)
        return ThumbnailsBatchResult(video_id=req.video_id, thumbnails=[], sprite_url=None, metadata={"status": "queued"})
    else:
        # synchronous
        local_results = []
        if req.timestamps:
            res_objs = gen.extract_at_timestamps(req.video_path, req.video_id, req.timestamps, specs)
        else:
            res_objs = gen.extract_evenly_spaced(req.video_path, req.video_id, count=req.count, specs=specs)
        local_paths = [r.url for r in res_objs]
        batch = _upload_and_build_result(req.video_id, local_paths)
        return batch


@app.get("/thumbnails/{video_id}", response_model=ThumbnailsBatchResult)
def list_thumbnails(video_id: str):
    # naive listing by directory
    dirp = os.path.join(config.OUTPUT_DIR, video_id)
    if not os.path.exists(dirp):
        raise HTTPException(status_code=404, detail="Not found")
    files = sorted([os.path.join(dirp, f) for f in os.listdir(dirp) if os.path.isfile(os.path.join(dirp,f))])
    return _upload_and_build_result(video_id, files)
