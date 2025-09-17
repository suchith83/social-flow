"""
Thumbnail generator for videos
"""

import os
from .utils import ensure_dir, run_command, logger
from .config import config


class Thumbnailer:
    def generate(self, input_file: str, video_id: str) -> str:
        out_dir = os.path.join(config.OUTPUT_DIR, video_id, "thumbnails")
        ensure_dir(out_dir)
        out_file = os.path.join(out_dir, "thumb.jpg")
        cmd = f"""{config.FFMPEG_PATH} -i {input_file} -ss 00:00:05 -vframes 1 {out_file}"""
        run_command(cmd)
        logger.info(f"Thumbnail generated for {video_id}")
        return out_file
