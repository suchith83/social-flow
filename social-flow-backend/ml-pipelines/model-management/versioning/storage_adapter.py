# LocalFS and S3-compatible storage adapters
# storage_adapter.py
"""
Storage adapters for model artifacts.
Supports:
 - Local filesystem adapter (default)
 - S3-compatible adapter using boto3 (optional)
This code avoids importing boto3 by default to keep deps minimal; import only when used.
"""

from typing import Protocol, BinaryIO, Optional
from pathlib import Path
import shutil
import os
import io


class StorageAdapter(Protocol):
    def upload(self, src_path: str, dest_path: str) -> str:
        """Upload file from src_path (local) to dest_path (storage-specific path) and return URI"""
        ...

    def download(self, uri: str, dest_path: str) -> str:
        """Download a file from storage to dest_path local path. Return local path"""
        ...

    def exists(self, uri: str) -> bool:
        ...

    def list(self, prefix: str):
        ...


class LocalStorageAdapter:
    """
    Stores artifacts under a root directory and exposes file:// URIs.
    """

    def __init__(self, root: str = "model_store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _full(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def upload(self, src_path: str, dest_path: str) -> str:
        dest = self._full(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
        return f"file://{dest.resolve()}"

    def download(self, uri: str, dest_path: str) -> str:
        # expect file://
        if uri.startswith("file://"):
            src = Path(uri[len("file://"):])
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            return str(dest.resolve())
        else:
            raise ValueError("Unsupported URI scheme for LocalStorageAdapter")

    def exists(self, uri: str) -> bool:
        if uri.startswith("file://"):
            return Path(uri[len("file://"):]).exists()
        return False

    def list(self, prefix: str):
        base = self._full(prefix)
        if not base.exists():
            return []
        return [str(p) for p in base.rglob("*") if p.is_file()]


class S3StorageAdapter:
    """
    Very small S3 adapter that uses boto3. Lazily import boto3.
    Requires AWS creds in environment or profile.
    uri format expected: s3://bucket/path/to/object
    """

    def __init__(self, aws_profile: Optional[str] = None):
        try:
            import boto3  # local import
        except Exception as e:
            raise RuntimeError("boto3 is required for S3StorageAdapter") from e
        self.boto3 = boto3
        self.session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        self.s3 = self.session.client("s3")

    def _parse(self, s3_uri: str):
        assert s3_uri.startswith("s3://")
        remainder = s3_uri[len("s3://"):]
        bucket, _, key = remainder.partition("/")
        return bucket, key

    def upload(self, src_path: str, dest_path: str) -> str:
        # dest_path like bucket/path
        if dest_path.startswith("s3://"):
            bucket, key = self._parse(dest_path)
        else:
            bucket, key = dest_path.split("/", 1)
        self.s3.upload_file(src_path, bucket, key)
        return f"s3://{bucket}/{key}"

    def download(self, uri: str, dest_path: str) -> str:
        bucket, key = self._parse(uri)
        self.s3.download_file(bucket, key, dest_path)
        return dest_path

    def exists(self, uri: str) -> bool:
        bucket, key = self._parse(uri)
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except self.boto3.exceptions.ClientError:
            return False

    def list(self, prefix: str):
        bucket, key_prefix = self._parse(prefix)
        paginator = self.s3.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                files.append(f"s3://{bucket}/{obj['Key']}")
        return files
