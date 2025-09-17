import pytest
from processed_videos.thumbnailer import Thumbnailer


def test_thumbnailer_init():
    th = Thumbnailer()
    assert th is not None
