"""
Utility functions for AWS S3 integration.
"""

import mimetypes


def guess_mime_type(filename: str) -> str:
    """Guess MIME type from filename."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def generate_presigned_url(client, key: str, expires_in: int = 3600):
    """Generate presigned URL for secure temporary access."""
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": client.bucket_name, "Key": key},
        ExpiresIn=expires_in,
    )
