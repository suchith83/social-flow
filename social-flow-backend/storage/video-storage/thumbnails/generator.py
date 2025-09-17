"""
Core thumbnail generator using ffmpeg.

Features:
- Extract multiple thumbnails at timestamp(s) or evenly spaced intervals
- Generate multiple sizes and formats
- Safe temporary file handling, and atomic moves into output dir
- Optional perceptual hashing integration (calls dedupe module)
"""

import os
import uuid
from typing import List, Optional, Tuple
from pathlib import Path

from .config import config
from .utils import ensure_dir, run_command, ffprobe_metadata, logger
from .models import ThumbnailSpec, ThumbnailResult
from .dedupe import compute_phash_if_enabled

TMP_DIR = os.path.join(config.OUTPUT_DIR, "tmp")
ensure_dir(TMP_DIR)
ensure_dir(config.OUTPUT_DIR)


def _size_str_to_tuple(size_str: str) -> Tuple[int, int]:
    w, h = size_str.split("x")
    return int(w), int(h)


def default_sizes() -> List[Tuple[int,int]]:
    sizes = []
    for s in config.DEFAULT_SIZES.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            sizes.append(_size_str_to_tuple(s))
        except Exception:
            logger.warning("Invalid DEFAULT_SIZES entry: %s", s)
    return sizes


class ThumbnailGenerator:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or config.OUTPUT_DIR
        ensure_dir(self.output_dir)

    def _build_output_path(self, video_id: str, thumb_id: str, size: Tuple[int,int], fmt: str) -> str:
        w, h = size
        sub = os.path.join(self.output_dir, video_id)
        ensure_dir(sub)
        return os.path.join(sub, f"{thumb_id}_{w}x{h}.{fmt}")

    def extract_at_timestamps(self, video_path: str, video_id: str, timestamps: List[float],
                              specs: List[ThumbnailSpec]) -> List[ThumbnailResult]:
        """
        Extract thumbnails at given timestamps (seconds).
        Returns list of ThumbnailResult objects.
        """
        results = []
        for ts_index, t in enumerate(timestamps):
            for spec_index, spec in enumerate(specs):
                thumb_id = uuid.uuid4().hex
                out = self._build_output_path(video_id, thumb_id, (spec.width, spec.height), spec.format)
                tmp_out = out + ".tmp"
                # FFmpeg command:
                # -ss before input is faster but less accurate; here we use -ss before -i with -noaccurate_seek avoided.
                cmd = [
                    config.FFMPEG_PATH,
                    "-ss", str(t),
                    "-i", video_path,
                    "-vframes", "1",
                    "-vf", f"scale=w={spec.width}:h={spec.height}:force_original_aspect_ratio=decrease,pad={spec.width}:{spec.height}:(ow-iw)/2:(oh-ih)/2",
                    "-q:v", str(max(2, int((100 - spec.quality)/2))),  # map quality to qscale roughly
                    "-f", "image2",
                    tmp_out
                ]
                run_command(cmd)
                os.replace(tmp_out, out)
                phash = compute_phash_if_enabled(out) if config.ENABLE_PHASH else None
                res = ThumbnailResult(
                    video_id=video_id,
                    thumbnail_id=thumb_id,
                    size=f"{spec.width}x{spec.height}",
                    format=spec.format,
                    url=out,
                    phash=phash,
                    width=spec.width,
                    height=spec.height
                )
                results.append(res)
        return results

    def extract_evenly_spaced(self, video_path: str, video_id: str, count: int = 5,
                              specs: Optional[List[ThumbnailSpec]] = None) -> List[ThumbnailResult]:
        """
        Extract `count` thumbnails evenly spaced across video's duration.
        """
        metadata = ffprobe_metadata(video_path)
        duration = None
        try:
            duration = float(metadata.get("format", {}).get("duration", 0.0))
        except Exception:
            duration = 0.0
        if duration <= 0:
            # fallback: grab first frames
            timestamps = [1.0 for _ in range(count)]
        else:
            timestamps = [max(0.5, duration * (i + 1) / (count + 1)) for i in range(count)]
        if specs is None:
            specs = [ThumbnailSpec(width=w, height=h) for (w,h) in default_sizes()[:config.MAX_THUMBNAILS]]
        return self.extract_at_timestamps(video_path, video_id, timestamps, specs)
