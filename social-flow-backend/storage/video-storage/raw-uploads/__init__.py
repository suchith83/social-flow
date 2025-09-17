"""
Raw Uploads module

Responsibilities:
- Accept raw video uploads (single-shot and chunked/resumable)
- Validate file type, size, and basic integrity
- Extract metadata for later processing
- Optionally scan uploads for viruses (stubbed)
- Provide presigned S3/MinIO upload URLs
- Stage files locally if direct upload is not used
- Publish events (hook points) for downstream systems (processor/analytics)
"""
