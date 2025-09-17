"""
Extract metadata from videos
"""

import subprocess
import json
from .config import config
from .utils import logger


class MetadataExtractor:
    def extract(self, input_file: str) -> dict:
        cmd = f"""ffprobe -v quiet -print_format json -show_format -show_streams {input_file}"""
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = proc.communicate()
        if proc.returncode != 0:
            logger.error(f"Metadata extraction failed: {err.decode()}")
            return {}
        return json.loads(out.decode())
