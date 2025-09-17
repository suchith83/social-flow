"""
Video Transcoder using FFmpeg
"""

import os
from .utils import run_command, ensure_dir, logger
from .config import config


class Transcoder:
    def __init__(self):
        ensure_dir(config.HLS_OUTPUT_DIR)
        ensure_dir(config.DASH_OUTPUT_DIR)

    def transcode(self, input_stream: str, stream_id: str):
        """Transcode to HLS and DASH with adaptive bitrates"""
        hls_out = os.path.join(config.HLS_OUTPUT_DIR, stream_id)
        dash_out = os.path.join(config.DASH_OUTPUT_DIR, stream_id)
        ensure_dir(hls_out)
        ensure_dir(dash_out)

        cmd = f"""{config.TRANSCODER_PATH} -i {input_stream} \
        -preset veryfast -g 48 -sc_threshold 0 \
        -map 0:v:0 -map 0:a:0 -c:v libx264 -c:a aac -b:v:3000k -b:a:128k \
        -f hls -hls_time 4 -hls_playlist_type event {hls_out}/index.m3u8 \
        -f dash {dash_out}/manifest.mpd"""

        run_command(cmd)
        logger.info(f"Stream {stream_id} transcoded to HLS & DASH")
