# Test Videos Fixtures

Place small sample video files here for local end-to-end tests that exercise upload/processing pipelines.

Recommended structure:
- data/fixtures/test-videos/
  - small.mp4       (10-20s sample)
  - tiny.webm       (short webm sample)
  - thumb.jpg       (expected thumbnail)

Notes:
- Do NOT commit large binaries to the repo. Instead, add small samples (< 2MB) or use Git LFS.
- Update service tests to reference files under this directory when running integration tests locally.
- For CI, prefer generating or downloading fixtures at runtime to avoid large repo size.
