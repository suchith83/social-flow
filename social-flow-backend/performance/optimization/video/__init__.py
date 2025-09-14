
"""
Video Optimization Package

Provides tools for optimizing video processing and delivery:
- Transcoding orchestration (FFmpeg wrapper)
- Adaptive streaming components (HLS/DASH segmenter simulation)
- Buffer (jitter buffer) management
- Codec selection and optimization heuristics
- Resolution / bitrate adaptation algorithms (ABR)
- Segment caching
- Monitoring & metrics collection
- Lightweight middleware for ASGI frameworks

Note: Some modules include glue code for calling external tools (ffmpeg). Ensure those binaries
are available in production and secure subprocess usage accordingly.
"""

from .transcoding import Transcoder, TranscodeJob
from .adaptive_streaming import AdaptiveStreamer, Segment
from .buffer_management import JitterBuffer
from .codec_optimization import CodecAdvisor
from .resolution_adaptation import ABRController
from .caching import SegmentCache
from .monitoring import VideoMetricsCollector
from .middleware import VideoOptimizationMiddleware

__all__ = [
    "Transcoder",
    "TranscodeJob",
    "AdaptiveStreamer",
    "Segment",
    "JitterBuffer",
    "CodecAdvisor",
    "ABRController",
    "SegmentCache",
    "VideoMetricsCollector",
    "VideoOptimizationMiddleware",
]
