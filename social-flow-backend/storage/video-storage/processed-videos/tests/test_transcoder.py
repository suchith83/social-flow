import pytest
from processed_videos.transcoder import VideoTranscoder


def test_transcoder_init():
    t = VideoTranscoder()
    assert t is not None
