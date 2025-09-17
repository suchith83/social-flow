"""
API layer for processed videos
"""

from fastapi import FastAPI, UploadFile
import uuid
import shutil
import os
from .processor import VideoProcessor
from .config import config

app = FastAPI(title="Processed Videos API")
processor = VideoProcessor()


@app.post("/process-video/")
async def process_video(file: UploadFile):
    video_id = str(uuid.uuid4())
    file_path = os.path.join(config.OUTPUT_DIR, f"{video_id}.mp4")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = processor.process(file_path, video_id)
    return result.dict()
