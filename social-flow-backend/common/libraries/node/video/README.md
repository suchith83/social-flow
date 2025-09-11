# common/libraries/node/video

Advanced Node.js video processing library.

## Features
- Transcoding (ffmpeg) with retry/backoff and presets
- Thumbnail extraction (ffmpeg + sharp)
- Metadata extraction (ffprobe)
- HLS packaging (ffmpeg-generated segments and playlists)
- Streaming helper for Range requests
- Orchestrator (processor) to prepare input, transcode, generate thumbnails, package HLS, and upload outputs to a storage adapter
- Express upload handler integration (uses busboy)
- Temp file management and cleanup utilities
- Robust logging and error classes

## Requirements
- `ffmpeg` and `ffprobe` binaries installed on the host and accessible via PATH. You can set `FFMPEG_PATH` / `FFPROBE_PATH` env vars if located elsewhere.
- Recommended Node packages:
