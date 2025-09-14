# Implements adaptive bitrate streaming
"""
adaptive_streaming.py

Simplified adaptive streaming utilities (HLS/DASH segment abstraction and a mock segmenter).
Production systems should use specialized tools (FFmpeg, Shaka Packager, Bento4) for packaging.

This module focuses on:
- Segment representation and metadata
- Lightweight segmenter that slices input into segments (simulated)
- Manifest generation helpers (HLS master and variant playlists)
"""

import os
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Segment:
    uri: str
    duration: float
    sequence: int
    byte_range: Optional[str] = None  # "start-end" if using byte-range
    timestamp: float = field(default_factory=time.time)


class AdaptiveStreamer:
    """
    Very small HLS manifest & segment manager (simulation).
    For real production, prefer invoking FFmpeg to generate segments or using packagers.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.segments: Dict[str, List[Segment]] = {}  # variant_name -> list of segments

    def add_segment(self, variant: str, filename: str, duration: float) -> Segment:
        seq = len(self.segments.get(variant, []))
        seg = Segment(uri=filename, duration=duration, sequence=seq)
        self.segments.setdefault(variant, []).append(seg)
        return seg

    def generate_variant_playlist(self, variant: str, target_duration: Optional[int] = None) -> str:
        segs = self.segments.get(variant, [])
        if not segs:
            return ""
        target_duration = target_duration or math.ceil(max(s.duration for s in segs))
        playlist_lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:0"
        ]
        for s in segs:
            playlist_lines.append(f"#EXTINF:{s.duration:.3f},")
            playlist_lines.append(s.uri)
        playlist_lines.append("#EXT-X-ENDLIST")
        return "\n".join(playlist_lines)

    def generate_master_playlist(self, variants: Dict[str, Dict]) -> str:
        """
        variants: {name: {bandwidth, resolution, uri}}
        """
        lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
        for name, conf in variants.items():
            bandwidth = conf.get("bandwidth", 0)
            resolution = conf.get("resolution", "")
            uri = conf.get("uri", "")
            lines.append(f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},RESOLUTION={resolution}')
            lines.append(uri)
        return "\n".join(lines)

    # Convenience: write playlists to disk
    def write_playlist(self, path: str, content: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
