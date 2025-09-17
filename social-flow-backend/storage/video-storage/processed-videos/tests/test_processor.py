import pytest
from processed_videos.processor import VideoProcessor


def test_processor_init():
    vp = VideoProcessor()
    assert vp is not None
