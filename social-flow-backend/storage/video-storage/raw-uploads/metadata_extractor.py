"""
Extract basic metadata (duration, codec, resolution) via ffprobe

Falls back to minimal data dict when ffprobe not available.
"""

import subprocess
import json
from typing import Dict
from .utils import logger


def ffprobe_metadata(path: str) -> Dict:
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(proc.stdout)
    except Exception as e:
        logger.warning(f"ffprobe failed for {path}: {e}")
        return {"error": "ffprobe-failed"}


def extract_metadata(path: str) -> Dict:
    md = ffprobe_metadata(path)
    # Normalize into friendly structure
    if "error" in md:
        return {"raw": md}
    fmt = md.get("format", {})
    streams = md.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return {
        "duration": float(fmt.get("duration")) if fmt.get("duration") else None,
        "size_bytes": int(fmt.get("size")) if fmt.get("size") else None,
        "bit_rate": int(fmt.get("bit_rate")) if fmt.get("bit_rate") else None,
        "video_codec": video_stream.get("codec_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "frame_rate": video_stream.get("r_frame_rate"),
        "audio_codec": audio_stream.get("codec_name"),
        "raw": md,
    }
