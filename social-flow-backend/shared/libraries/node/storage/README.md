# common/libraries/node/storage

Production-ready storage abstraction library for Node.js.

## Features
- Adapter-driven: Local FS adapter for dev, S3 adapter for production (S3-compatible)
- Streaming uploads & downloads
- Multipart support (leveraging AWS SDK lib-storage when available)
- Presigned URLs (S3) for direct client uploads/downloads
- Safe local storage with metadata sidecars
- Upload helpers with dedupe and quota checks
- Lifecycle utilities (scan usage, expire, garbage-collect)
- Robust retry/backoff and transient error handling
- Clean error types and logging

## Install
For S3 adapter:
```bash
npm install @aws-sdk/client-s3 @aws-sdk/lib-storage @aws-sdk/s3-request-presigner
