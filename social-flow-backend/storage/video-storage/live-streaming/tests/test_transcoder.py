import pytest
from live_streaming.transcoder import Transcoder


def test_transcoder_init():
    transcoder = Transcoder()
    assert transcoder is not None
