import pytest
from live_streaming.packager import Packager


def test_packager():
    p = Packager()
    assert p.get_hls_playlist("nonexistent") == ""
