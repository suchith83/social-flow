"""
Transcode videos into multiple resolutions using FFmpeg
"""

import os
from .utils import ensure_dir, run_command, logger
from .config import config


class VideoTranscoder:
    def __init__(self):
        ensure_dir(config.OUTPUT_DIR)

    def transcode(self, input_file: str, video_id: str) -> dict:
        """Generate multiple resolutions (1080p, 720p, 480p)"""
        out_dir = os.path.join(config.OUTPUT_DIR, video_id, "resolutions")
        ensure_dir(out_dir)

        resolutions = {
            "1080p": "1920x1080",
            "720p": "1280x720",
            "480p": "854x480",
        }
        outputs = {}

        for label, size in resolutions.items():
            out_path = os.path.join(out_dir, f"{label}.mp4")
            cmd = f"""{config.FFMPEG_PATH} -i {input_file} -vf scale={size} -c:v libx264 -c:a aac -strict -2 {out_path}"""
            run_command(cmd)
            outputs[label] = out_path
            logger.info(f"Generated {label} for {video_id}")

        return outputs
