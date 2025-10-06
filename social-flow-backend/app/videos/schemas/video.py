"""Compatibility re-export module.

Some legacy and new tests import VideoCreate (and potentially other schemas)
from `app.videos.schemas.video`. The canonical location for video schemas
was consolidated into `app.schemas.video`. To avoid changing widespread
imports (and to allow a gradual migration), this lightweight adapter
re-exports the public schema classes.

Remove this file once all imports are updated to: `from app.schemas.video import VideoCreate`.
"""
from app.schemas.video import (
    VideoBase,
    VideoCreate,
    VideoUploadInit,
    VideoUploadComplete,
    VideoUpdate,
    VideoResponse,
    VideoDetailResponse,
    VideoPublicResponse,
    VideoUploadURL,
    VideoStreamingURLs,
    VideoAnalytics,
    VideoListFilters,
    VideoSortOptions,
    VideoBatchUpdate,
    VideoBatchDelete,
    VideoStatus,
    VideoVisibility,
    VideoQuality,
)

__all__ = [
    "VideoBase",
    "VideoCreate",
    "VideoUploadInit",
    "VideoUploadComplete",
    "VideoUpdate",
    "VideoResponse",
    "VideoDetailResponse",
    "VideoPublicResponse",
    "VideoUploadURL",
    "VideoStreamingURLs",
    "VideoAnalytics",
    "VideoListFilters",
    "VideoSortOptions",
    "VideoBatchUpdate",
    "VideoBatchDelete",
    "VideoStatus",
    "VideoVisibility",
    "VideoQuality",
]
