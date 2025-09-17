"""
Main pipeline that orchestrates video processing
"""

import datetime
from .transcoder import VideoTranscoder
from .thumbnailer import Thumbnailer
from .metadata_extractor import MetadataExtractor
from .quality_analyzer import QualityAnalyzer
from .storage_uploader import StorageUploader
from .models import ProcessedVideo


class VideoProcessor:
    def __init__(self):
        self.transcoder = VideoTranscoder()
        self.thumbnailer = Thumbnailer()
        self.metadata_extractor = MetadataExtractor()
        self.quality_analyzer = QualityAnalyzer()
        self.uploader = StorageUploader()

    def process(self, input_file: str, video_id: str) -> ProcessedVideo:
        resolutions = self.transcoder.transcode(input_file, video_id)
        thumbnail = self.thumbnailer.generate(input_file, video_id)
        metadata = self.metadata_extractor.extract(input_file)
        quality = self.quality_analyzer.analyze(input_file)

        urls = []
        for _, f in resolutions.items():
            urls.append(self.uploader.upload(f, video_id))
        urls.append(self.uploader.upload(thumbnail, video_id))

        return ProcessedVideo(
            video_id=video_id,
            resolutions=list(resolutions.keys()),
            thumbnails=[thumbnail],
            metadata=metadata,
            quality=quality,
            storage_urls=urls,
            processed_at=datetime.datetime.utcnow(),
        )
