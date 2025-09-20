"""High-level video storage helpers used by video-service and workflows."""
from typing import BinaryIO, Optional, Dict
import os
import io
import time

from .s3_client import S3Client
from .filesystem import save_stream, read_stream

# get defaults from env
DEFAULT_BUCKET = os.environ.get("VIDEO_S3_BUCKET", "sf-videos")
PLAYBACK_EXPIRATION = int(os.environ.get("VIDEO_PLAYBACK_EXPIRE", "3600"))

# initialize client (uses boto3 if available, otherwise local fallback)
_s3 = S3Client(os.environ.get("S3_ENDPOINT"))


def upload_video(fileobj: BinaryIO, filename: str, bucket: Optional[str] = None) -> Dict[str, str]:
    """
    Uploads a video file-like object to storage and returns metadata containing
    the storage key and playback URL.
    """
    bucket = bucket or DEFAULT_BUCKET
    # create deterministic key: videos/<timestamp>_<filename>
    key = f"videos/{int(time.time())}_{os.path.basename(filename)}"
    # ensure fileobj is at start
    try:
        fileobj.seek(0)
    except Exception:
        pass
    # If s3 fallback writes to local disk, it's handled by S3Client
    _s3.upload_fileobj(fileobj, bucket, key)
    url = _s3.generate_presigned_url(bucket, key, expiration=PLAYBACK_EXPIRATION)
    return {"bucket": bucket, "key": key, "playback_url": url}


def get_playback_url(bucket: str, key: str, expires_in: Optional[int] = None) -> str:
    return _s3.generate_presigned_url(bucket, key, expires_in or PLAYBACK_EXPIRATION)


def download_video_to_bytes(bucket: str, key: str) -> bytes:
    buf = io.BytesIO()
    _s3.download_fileobj(bucket, key, buf)
    return buf.getvalue()


def delete_video(bucket: str, key: str) -> None:
    _s3.delete_object(bucket, key)
