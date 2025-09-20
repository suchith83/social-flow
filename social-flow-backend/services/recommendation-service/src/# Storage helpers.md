# Storage helpers

This package provides a small storage abstraction used by services:

- storage.s3_client.S3Client — prefers boto3 S3 client, falls back to a local filesystem DummyS3Client writing under `.storage_s3/`.
- storage.filesystem — utilities for saving/reading/removing files under `.local_storage/`.
- storage.video_storage — high-level helpers for uploading videos, getting playback URLs and deleting videos.

Local development
- By default the package will use the local fallback if boto3 is not installed.
- To test S3-compatible endpoints (MinIO), set S3_ENDPOINT env var to your MinIO HTTP endpoint and ensure boto3 is installed.

Example
```py
from storage.video_storage import upload_video
with open("sample.mp4","rb") as f:
    meta = upload_video(f, "sample.mp4")
print(meta["playback_url"])
```

Notes
- The DummyS3Client is a development convenience and is not suitable for production.
- For production, ensure boto3 is installed and S3 credentials are available in the environment or via IAM roles.
