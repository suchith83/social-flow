"""
Thumbnails Module

Responsibilities:
- Extract single and multiple thumbnails from video files
- Produce different sizes and formats (jpeg, webp)
- Generate contact sheets (grid) and sprite images
- Compute perceptual hashes (pHash) for deduplication and similarity checks
- Upload thumbnails to object storage (S3/MinIO) or local storage
- Provide async job hooks for background workers
- Expose REST endpoints to request thumbnails and retrieve metadata
"""
