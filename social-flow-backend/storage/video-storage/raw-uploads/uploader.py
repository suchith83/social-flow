"""
High-level uploader orchestration:
- init_upload: returns upload_id and chunk_size
- accept_chunk: writes chunk by index
- finalize_upload: assembles, validates, scans, uploads to backend, emits UploadCompleteEvent
"""

import os
import uuid
import logging
from typing import Optional
from .chunk_manager import (
    make_upload_id,
    init_upload,
    write_chunk,
    is_complete,
    assemble,
    cleanup_upload,
)
from .models import InitUploadResponse, InitUploadRequest, UploadCompleteEvent
from .config import config
from .storage_backend import StorageBackend
from .validator import validate_staged_file, UploadValidationError
from .antivirus import scan_file, VirusDetected
from .metadata_extractor import extract_metadata
from .utils import ensure_dir, logger

storage = StorageBackend()


def init_upload_flow(req: InitUploadRequest, chunk_size: int = 5 * 1024 * 1024) -> InitUploadResponse:
    upload_id = make_upload_id()
    init_upload(upload_id, req.filename, req.total_bytes, chunk_size)
    # presign optionally if using direct S3
    presigned = None
    if config.STORAGE_BACKEND == "s3":
        key = f"{upload_id}/{req.filename}"
        presigned = storage.presign_upload(key, expires_in=config.PRESIGNED_URL_TTL)
        upload_url = presigned["url"]
    else:
        upload_url = None
    return InitUploadResponse(upload_id=upload_id, chunk_size=chunk_size, upload_url=upload_url, expires_in=config.PRESIGNED_URL_TTL)


def accept_chunk_flow(upload_id: str, chunk_index: int, data: bytes) -> int:
    # basic quick checks could be added here (e.g., chunk size)
    bytes_written = write_chunk(upload_id, chunk_index, data)
    # emit a ChunkCompleteEvent hook here if needed
    return bytes_written


def finalize_upload_flow(upload_id: str, user_id: Optional[str] = None) -> UploadCompleteEvent:
    if not is_complete(upload_id):
        raise RuntimeError("Upload not complete")
    staged_path = assemble(upload_id)
    # Validation
    try:
        validate_staged_file(staged_path)
    except UploadValidationError as e:
        cleanup_upload(upload_id)
        raise
    # Antivirus scan
    try:
        scan_file(staged_path)
    except VirusDetected:
        cleanup_upload(upload_id)
        raise
    # extract metadata
    md = extract_metadata(staged_path)
    # upload to backend
    key = f"{upload_id}/{os.path.basename(staged_path)}"
    external_url = storage.upload_file(staged_path, key)
    # publish event
    event = UploadCompleteEvent(upload_id=upload_id, file_path=external_url, total_bytes=os.path.getsize(staged_path), user_id=user_id)
    # cleanup local chunks
    cleanup_upload(upload_id)
    return event
