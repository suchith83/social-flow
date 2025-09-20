"""S3 client wrapper with a local filesystem fallback for development and tests."""
from typing import BinaryIO, Optional
import os
import io
import errno

# logger fallback: try common, then src.utils, then stdlib
try:
    from common.libraries.python.monitoring.logger import get_logger  # type: ignore
except Exception:
    try:
        from src.utils import get_logger  # type: ignore
    except Exception:
        import logging

        def get_logger(name: str):
            l = logging.getLogger(name)
            if not l.handlers:
                l.addHandler(logging.StreamHandler())
            l.setLevel(logging.INFO)
            return l

logger = get_logger("storage.s3_client")

# Try boto3
try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_S3_PREFIX = os.environ.get("LOCAL_S3_PREFIX", os.path.join(ROOT, ".storage_s3"))


class DummyS3Client:
    """Local filesystem-based emulation of minimal S3 behaviors."""

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or LOCAL_S3_PREFIX
        os.makedirs(self.base_path, exist_ok=True)

    def _path(self, bucket: str, key: str) -> str:
        full = os.path.join(self.base_path, bucket, key)
        dirpath = os.path.dirname(full)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        return full

    def upload_fileobj(self, fileobj: BinaryIO, bucket: str, key: str) -> None:
        dst = self._path(bucket, key)
        logger.debug("DummyS3Client.upload_fileobj -> %s", dst)
        # Read in chunks to support file-like objects
        with open(dst, "wb") as f:
            fileobj.seek(0)
            while True:
                chunk = fileobj.read(8192)
                if not chunk:
                    break
                f.write(chunk)

    def download_fileobj(self, bucket: str, key: str, fileobj: BinaryIO) -> None:
        src = self._path(bucket, key)
        logger.debug("DummyS3Client.download_fileobj <- %s", src)
        with open(src, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                fileobj.write(chunk)
            fileobj.seek(0)

    def generate_presigned_url(self, bucket: str, key: str, expiration: int = 3600) -> str:
        path = self._path(bucket, key)
        # For local fallback return a file:// URL
        return f"file://{os.path.abspath(path)}"

    def delete_object(self, bucket: str, key: str) -> None:
        p = self._path(bucket, key)
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


class S3Client:
    """Thin wrapper that uses boto3 if available, otherwise DummyS3Client."""

    def __init__(self, aws_url: Optional[str] = None):
        if boto3 is not None:
            try:
                # aws_url can be a region or endpoint_url; keep init flexible
                if aws_url:
                    # if aws_url looks like an endpoint_url (http), pass it through
                    if aws_url.startswith("http"):
                        self._client = boto3.client("s3", endpoint_url=aws_url)
                    else:
                        self._client = boto3.client("s3")
                else:
                    self._client = boto3.client("s3")
                logger.info("Initialized real S3 client")
                self._use_dummy = False
            except Exception:
                logger.exception("Failed to initialize boto3 client; using DummyS3Client")
                self._client = DummyS3Client()
                self._use_dummy = True
        else:
            logger.info("boto3 not available; using DummyS3Client")
            self._client = DummyS3Client()
            self._use_dummy = True

    def upload_fileobj(self, fileobj: BinaryIO, bucket: str, key: str) -> None:
        if self._use_dummy:
            return self._client.upload_fileobj(fileobj, bucket, key)
        try:
            # boto3 expects fileobj and bucket/key params
            self._client.upload_fileobj(Fileobj=fileobj, Bucket=bucket, Key=key)
        except (BotoCoreError, ClientError, Exception):
            logger.exception("S3 upload failed; falling back to DummyS3Client")
            # fallback write to disk using dummy
            DummyS3Client().upload_fileobj(fileobj, bucket, key)

    def download_fileobj(self, bucket: str, key: str, fileobj: BinaryIO) -> None:
        if self._use_dummy:
            return self._client.download_fileobj(bucket, key, fileobj)
        try:
            self._client.download_fileobj(Bucket=bucket, Key=key, Fileobj=fileobj)
        except (BotoCoreError, ClientError, Exception):
            logger.exception("S3 download failed; attempting DummyS3Client")
            DummyS3Client().download_fileobj(bucket, key, fileobj)

    def generate_presigned_url(self, bucket: str, key: str, expiration: int = 3600) -> str:
        if self._use_dummy:
            return self._client.generate_presigned_url(bucket, key, expiration)
        try:
            return self._client.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expiration
            )
        except (BotoCoreError, ClientError, Exception):
            logger.exception("generate_presigned_url failed; returning file path")
            return DummyS3Client().generate_presigned_url(bucket, key, expiration)

    def delete_object(self, bucket: str, key: str) -> None:
        if self._use_dummy:
            return self._client.delete_object(bucket, key)
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
        except Exception:
            logger.exception("S3 delete failed; attempting DummyS3Client")
            DummyS3Client().delete_object(bucket, key)
