"""
FastAPI endpoints for raw uploads

Endpoints:
- POST /init-upload -> initialize an upload (returns upload_id, chunk_size, optional presigned URL)
- PUT /upload/{upload_id}/chunk/{index} -> upload chunk bytes (server-assisted)
- POST /complete/{upload_id} -> finalize (assemble + validate + upload)
- GET /status/{upload_id} -> show manifest/chunk list
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uvicorn
import os

from .models import InitUploadRequest
from .uploader import init_upload_flow, accept_chunk_flow, finalize_upload_flow
from .chunk_manager import list_received_chunks, manifest_path
from .utils import ensure_dir, logger

app = FastAPI(title="Raw Uploads API")


@app.post("/init-upload")
async def init_upload(
    filename: str = Form(...),
    total_bytes: int = Form(...),
    mime_type: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
):
    req = InitUploadRequest(filename=filename, total_bytes=total_bytes, mime_type=mime_type, user_id=user_id)
    resp = init_upload_flow(req)
    return resp.dict()


@app.put("/upload/{upload_id}/chunk/{index}")
async def upload_chunk(upload_id: str, index: int, chunk: UploadFile = File(...)):
    # read bytes (careful with memory for large chunks; FastAPI's UploadFile is file-like)
    data = await chunk.read()
    try:
        written = accept_chunk_flow(upload_id, index, data)
        return JSONResponse({"bytes_written": written})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/complete/{upload_id}")
def complete_upload(upload_id: str, user_id: Optional[str] = None):
    try:
        event = finalize_upload_flow(upload_id, user_id=user_id)
        return event.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/status/{upload_id}")
def status(upload_id: str):
    mpath = manifest_path(upload_id)
    if not os.path.exists(mpath):
        raise HTTPException(status_code=404, detail="Upload not found")
    with open(mpath, "r") as f:
        import json
        return json.load(f)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
