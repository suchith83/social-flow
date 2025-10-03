"""
Advanced Video Encoding Service with AWS MediaConvert Integration.

This module handles video transcoding to multiple formats and quality levels
with HLS/DASH adaptive streaming output. Supports both cloud (MediaConvert)
and local (FFmpeg) encoding with automatic fallback.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import boto3
from botocore.exceptions import ClientError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import VideoServiceError
from app.infrastructure.storage.s3_backend import S3StorageBackend
from app.videos.models.encoding_job import EncodingJob, EncodingStatus

logger = logging.getLogger(__name__)


class EncodingPreset(str, Enum):
    """Video encoding quality presets."""
    MOBILE_240P = "240p"
    SD_360P = "360p"
    SD_480P = "480p"
    HD_720P = "720p"
    FULL_HD_1080P = "1080p"
    UHD_4K = "4k"


class VideoCodec(str, Enum):
    """Supported video codecs."""
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"


class AudioCodec(str, Enum):
    """Supported audio codecs."""
    AAC = "aac"
    MP3 = "mp3"
    OPUS = "opus"


# Quality settings for each preset
QUALITY_PRESETS = {
    EncodingPreset.MOBILE_240P: {
        "width": 426,
        "height": 240,
        "video_bitrate": "400k",
        "audio_bitrate": "64k",
        "framerate": 24,
    },
    EncodingPreset.SD_360P: {
        "width": 640,
        "height": 360,
        "video_bitrate": "800k",
        "audio_bitrate": "96k",
        "framerate": 30,
    },
    EncodingPreset.SD_480P: {
        "width": 854,
        "height": 480,
        "video_bitrate": "1400k",
        "audio_bitrate": "128k",
        "framerate": 30,
    },
    EncodingPreset.HD_720P: {
        "width": 1280,
        "height": 720,
        "video_bitrate": "2800k",
        "audio_bitrate": "192k",
        "framerate": 30,
    },
    EncodingPreset.FULL_HD_1080P: {
        "width": 1920,
        "height": 1080,
        "video_bitrate": "5000k",
        "audio_bitrate": "192k",
        "framerate": 30,
    },
    EncodingPreset.UHD_4K: {
        "width": 3840,
        "height": 2160,
        "video_bitrate": "15000k",
        "audio_bitrate": "256k",
        "framerate": 30,
    },
}


class AdvancedVideoEncodingService:
    """
    Advanced video encoding service with AWS MediaConvert and FFmpeg support.
    
    Features:
    - Multi-quality encoding (240p to 4K)
    - HLS and DASH adaptive streaming
    - AWS MediaConvert integration (cloud)
    - FFmpeg fallback (local)
    - Progress tracking
    - Thumbnail generation
    - Error handling and retry logic
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize encoding service.
        
        Args:
            db: Database session for tracking encoding jobs
        """
        self.db = db
        self.storage = S3StorageBackend()
        
        # Initialize AWS MediaConvert client
        try:
            self.mediaconvert_client = boto3.client(
                "mediaconvert",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )
            
            # Get MediaConvert endpoint
            endpoints = self.mediaconvert_client.describe_endpoints()
            self.mediaconvert_endpoint = endpoints["Endpoints"][0]["Url"]
            
            # Create client with custom endpoint
            self.mediaconvert_client = boto3.client(
                "mediaconvert",
                region_name=settings.AWS_REGION,
                endpoint_url=self.mediaconvert_endpoint,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )
            
            self.use_mediaconvert = True
            logger.info(f"AWS MediaConvert initialized: {self.mediaconvert_endpoint}")
            
        except Exception as e:
            logger.warning(f"MediaConvert not available, using FFmpeg fallback: {e}")
            self.use_mediaconvert = False
            self.mediaconvert_client = None

    async def encode_video(
        self,
        video_id: UUID,
        input_path: str,
        qualities: Optional[List[EncodingPreset]] = None,
        output_format: str = "hls",
    ) -> Dict[str, any]:
        """
        Encode video to multiple quality levels.
        
        Args:
            video_id: ID of the video to encode
            input_path: S3 key or local path to input video
            qualities: List of quality presets to encode (default: all)
            output_format: Output format - 'hls' or 'dash'
        
        Returns:
            Dictionary with encoding job details and output paths
        """
        if qualities is None:
            # Default qualities based on video resolution
            qualities = [
                EncodingPreset.MOBILE_240P,
                EncodingPreset.SD_480P,
                EncodingPreset.HD_720P,
                EncodingPreset.FULL_HD_1080P,
            ]

        # Create encoding job record
        encoding_job = EncodingJob(
            video_id=video_id,
            status=EncodingStatus.PENDING,
            input_path=input_path,
            output_format=output_format,
            started_at=datetime.now(timezone.utc),
        )
        
        self.db.add(encoding_job)
        await self.db.commit()
        await self.db.refresh(encoding_job)

        try:
            # Update status to processing
            encoding_job.status = EncodingStatus.PROCESSING
            await self.db.commit()

            # Choose encoding method
            if self.use_mediaconvert and settings.AWS_ACCESS_KEY_ID:
                result = await self._encode_with_mediaconvert(
                    video_id,
                    input_path,
                    qualities,
                    output_format,
                    encoding_job.id,
                )
            else:
                result = await self._encode_with_ffmpeg(
                    video_id,
                    input_path,
                    qualities,
                    output_format,
                    encoding_job.id,
                )

            # Update job with success
            encoding_job.status = EncodingStatus.COMPLETED
            encoding_job.completed_at = datetime.now(timezone.utc)
            encoding_job.output_paths = result["outputs"]
            encoding_job.hls_manifest_url = result.get("hls_manifest")
            encoding_job.dash_manifest_url = result.get("dash_manifest")
            
            await self.db.commit()

            logger.info(f"Video encoding completed: {video_id}")
            return result

        except Exception as e:
            # Update job with failure
            encoding_job.status = EncodingStatus.FAILED
            encoding_job.error_message = str(e)
            encoding_job.completed_at = datetime.now(timezone.utc)
            await self.db.commit()

            logger.error(f"Video encoding failed for {video_id}: {e}")
            raise VideoServiceError(f"Video encoding failed: {str(e)}")

    async def _encode_with_mediaconvert(
        self,
        video_id: UUID,
        input_path: str,
        qualities: List[EncodingPreset],
        output_format: str,
        job_id: UUID,
    ) -> Dict[str, any]:
        """
        Encode video using AWS MediaConvert (cloud).
        
        Args:
            video_id: Video ID
            input_path: S3 input path
            qualities: Quality presets
            output_format: Output format (hls/dash)
            job_id: Encoding job ID
        
        Returns:
            Dictionary with job details and output paths
        """
        # Construct S3 input and output paths
        s3_input = f"s3://{settings.S3_BUCKET_NAME}/{input_path}"
        s3_output_base = f"s3://{settings.S3_BUCKET_NAME}/videos/{video_id}/encoded/"

        # Build MediaConvert job settings
        job_settings = {
            "OutputGroups": self._build_mediaconvert_output_groups(
                qualities, output_format, s3_output_base
            ),
            "Inputs": [
                {
                    "FileInput": s3_input,
                    "AudioSelectors": {
                        "Audio Selector 1": {
                            "DefaultSelection": "DEFAULT"
                        }
                    },
                    "VideoSelector": {},
                    "TimecodeSource": "ZEROBASED",
                }
            ],
        }

        # Create MediaConvert job
        try:
            response = self.mediaconvert_client.create_job(
                Role=os.getenv("MEDIACONVERT_ROLE_ARN", ""),  # IAM role for MediaConvert
                Settings=job_settings,
                UserMetadata={
                    "video_id": str(video_id),
                    "encoding_job_id": str(job_id),
                },
            )

            mediaconvert_job_id = response["Job"]["Id"]
            logger.info(f"MediaConvert job created: {mediaconvert_job_id}")

            # Poll for job completion (in production, use EventBridge)
            await self._wait_for_mediaconvert_job(mediaconvert_job_id)

            # Build result
            return {
                "mediaconvert_job_id": mediaconvert_job_id,
                "outputs": self._generate_output_paths(video_id, qualities, output_format),
                "hls_manifest": f"{s3_output_base}master.m3u8" if output_format == "hls" else None,
                "dash_manifest": f"{s3_output_base}manifest.mpd" if output_format == "dash" else None,
            }

        except ClientError as e:
            logger.error(f"MediaConvert job creation failed: {e}")
            raise VideoServiceError(f"MediaConvert error: {str(e)}")

    def _build_mediaconvert_output_groups(
        self, qualities: List[EncodingPreset], output_format: str, s3_output_base: str
    ) -> List[Dict]:
        """Build MediaConvert output group configuration."""
        output_groups = []

        if output_format == "hls":
            # HLS adaptive streaming
            outputs = []
            for quality in qualities:
                preset = QUALITY_PRESETS[quality]
                outputs.append({
                    "NameModifier": f"_{quality.value}",
                    "VideoDescription": {
                        "Width": preset["width"],
                        "Height": preset["height"],
                        "CodecSettings": {
                            "Codec": "H_264",
                            "H264Settings": {
                                "Bitrate": int(preset["video_bitrate"].replace("k", "000")),
                                "RateControlMode": "CBR",
                                "CodecProfile": "MAIN",
                                "FramerateControl": "SPECIFIED",
                                "FramerateNumerator": preset["framerate"],
                                "FramerateDenominator": 1,
                            }
                        }
                    },
                    "AudioDescriptions": [{
                        "CodecSettings": {
                            "Codec": "AAC",
                            "AacSettings": {
                                "Bitrate": int(preset["audio_bitrate"].replace("k", "000")),
                                "CodecProfile": "LC",
                                "SampleRate": 48000,
                            }
                        }
                    }],
                    "ContainerSettings": {
                        "Container": "M3U8",
                        "M3u8Settings": {
                            "AudioFramesPerPes": 4,
                            "PcrControl": "PCR_EVERY_PES_PACKET",
                        }
                    },
                })

            output_groups.append({
                "Name": "HLS Adaptive Streaming",
                "OutputGroupSettings": {
                    "Type": "HLS_GROUP_SETTINGS",
                    "HlsGroupSettings": {
                        "Destination": s3_output_base,
                        "SegmentLength": 6,
                        "MinSegmentLength": 0,
                        "ManifestDurationFormat": "INTEGER",
                        "StreamInfResolution": "INCLUDE",
                        "CodecSpecification": "RFC_4281",
                    }
                },
                "Outputs": outputs,
            })

        elif output_format == "dash":
            # DASH adaptive streaming
            outputs = []
            for quality in qualities:
                preset = QUALITY_PRESETS[quality]
                outputs.append({
                    "NameModifier": f"_{quality.value}",
                    "VideoDescription": {
                        "Width": preset["width"],
                        "Height": preset["height"],
                        "CodecSettings": {
                            "Codec": "H_264",
                            "H264Settings": {
                                "Bitrate": int(preset["video_bitrate"].replace("k", "000")),
                                "RateControlMode": "CBR",
                            }
                        }
                    },
                    "AudioDescriptions": [{
                        "CodecSettings": {
                            "Codec": "AAC",
                            "AacSettings": {
                                "Bitrate": int(preset["audio_bitrate"].replace("k", "000")),
                            }
                        }
                    }],
                    "ContainerSettings": {
                        "Container": "MPD",
                    },
                })

            output_groups.append({
                "Name": "DASH Adaptive Streaming",
                "OutputGroupSettings": {
                    "Type": "DASH_ISO_GROUP_SETTINGS",
                    "DashIsoGroupSettings": {
                        "Destination": s3_output_base,
                        "SegmentLength": 6,
                        "FragmentLength": 2,
                    }
                },
                "Outputs": outputs,
            })

        return output_groups

    async def _wait_for_mediaconvert_job(self, job_id: str, timeout: int = 3600) -> None:
        """
        Poll MediaConvert job until completion.
        
        Args:
            job_id: MediaConvert job ID
            timeout: Maximum time to wait in seconds
        """
        start_time = datetime.now(timezone.utc)
        
        while True:
            try:
                response = self.mediaconvert_client.get_job(Id=job_id)
                status = response["Job"]["Status"]

                if status == "COMPLETE":
                    logger.info(f"MediaConvert job completed: {job_id}")
                    return
                
                elif status in ["ERROR", "CANCELED"]:
                    error_msg = response["Job"].get("ErrorMessage", "Unknown error")
                    raise VideoServiceError(f"MediaConvert job failed: {error_msg}")

                # Check timeout
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if elapsed > timeout:
                    raise VideoServiceError(f"MediaConvert job timeout: {job_id}")

                # Wait before next poll
                await asyncio.sleep(10)

            except ClientError as e:
                logger.error(f"Error checking MediaConvert job status: {e}")
                raise

    async def _encode_with_ffmpeg(
        self,
        video_id: UUID,
        input_path: str,
        qualities: List[EncodingPreset],
        output_format: str,
        job_id: UUID,
    ) -> Dict[str, any]:
        """
        Encode video using FFmpeg (local fallback).
        
        Args:
            video_id: Video ID
            input_path: Input file path
            qualities: Quality presets
            output_format: Output format (hls/dash)
            job_id: Encoding job ID
        
        Returns:
            Dictionary with encoding results
        """
        logger.info(f"Encoding video {video_id} with FFmpeg (local)")

        # Create temporary directory for processing
        temp_dir = Path(f"/tmp/encoding/{video_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Download video from S3 if needed
        if input_path.startswith("s3://") or "/" in input_path:
            local_input = temp_dir / "input.mp4"
            await self.storage.download(input_path, str(local_input))
        else:
            local_input = Path(input_path)

        outputs = {}
        
        try:
            if output_format == "hls":
                # Generate HLS adaptive streaming with FFmpeg
                master_playlist = temp_dir / "master.m3u8"
                variant_playlists = []

                for quality in qualities:
                    preset = QUALITY_PRESETS[quality]
                    variant_file = temp_dir / f"variant_{quality.value}.m3u8"
                    
                    # FFmpeg command for HLS encoding
                    cmd = [
                        "ffmpeg",
                        "-i", str(local_input),
                        "-c:v", "libx264",
                        "-preset", "medium",
                        "-b:v", preset["video_bitrate"],
                        "-vf", f"scale={preset['width']}:{preset['height']}",
                        "-r", str(preset["framerate"]),
                        "-c:a", "aac",
                        "-b:a", preset["audio_bitrate"],
                        "-hls_time", "6",
                        "-hls_playlist_type", "vod",
                        "-hls_segment_filename", str(temp_dir / f"segment_{quality.value}_%03d.ts"),
                        str(variant_file),
                    ]

                    # Execute FFmpeg
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        raise VideoServiceError(f"FFmpeg encoding failed: {stderr.decode()}")

                    variant_playlists.append((quality, variant_file))
                    outputs[quality.value] = f"videos/{video_id}/encoded/{variant_file.name}"

                # Create master playlist
                self._create_hls_master_playlist(master_playlist, variant_playlists)

                # Upload all files to S3
                await self._upload_encoded_files(video_id, temp_dir)

                return {
                    "ffmpeg": True,
                    "outputs": outputs,
                    "hls_manifest": f"videos/{video_id}/encoded/master.m3u8",
                    "dash_manifest": None,
                }

            else:
                raise VideoServiceError("DASH encoding with FFmpeg not yet implemented")

        finally:
            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_hls_master_playlist(
        self, master_file: Path, variants: List[Tuple[EncodingPreset, Path]]
    ) -> None:
        """Create HLS master playlist."""
        with open(master_file, "w") as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n\n")

            for quality, variant_path in variants:
                preset = QUALITY_PRESETS[quality]
                bandwidth = int(preset["video_bitrate"].replace("k", "000"))
                
                f.write(f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},")
                f.write(f"RESOLUTION={preset['width']}x{preset['height']}\n")
                f.write(f"{variant_path.name}\n")

    async def _upload_encoded_files(self, video_id: UUID, directory: Path) -> None:
        """Upload all encoded files to S3."""
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                s3_key = f"videos/{video_id}/encoded/{file_path.name}"
                with open(file_path, "rb") as f:
                    await self.storage.upload(
                        data=f.read(),
                        key=s3_key,
                        content_type=self._get_content_type(file_path.suffix),
                    )

    def _get_content_type(self, extension: str) -> str:
        """Get MIME type for file extension."""
        content_types = {
            ".m3u8": "application/vnd.apple.mpegurl",
            ".ts": "video/mp2t",
            ".mpd": "application/dash+xml",
            ".mp4": "video/mp4",
        }
        return content_types.get(extension, "application/octet-stream")

    def _generate_output_paths(
        self, video_id: UUID, qualities: List[EncodingPreset], output_format: str
    ) -> Dict[str, str]:
        """Generate output path dictionary."""
        return {
            quality.value: f"videos/{video_id}/encoded/{quality.value}/"
            for quality in qualities
        }

    async def generate_thumbnails(
        self, video_id: UUID, input_path: str, count: int = 5
    ) -> List[str]:
        """
        Generate thumbnails from video.
        
        Args:
            video_id: Video ID
            input_path: Input video path
            count: Number of thumbnails to generate
        
        Returns:
            List of S3 keys for generated thumbnails
        """
        logger.info(f"Generating {count} thumbnails for video {video_id}")

        temp_dir = Path(f"/tmp/thumbnails/{video_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Download video
        local_input = temp_dir / "input.mp4"
        if input_path.startswith("s3://") or "/" in input_path:
            await self.storage.download(input_path, str(local_input))
        else:
            local_input = Path(input_path)

        try:
            # Get video duration
            duration_cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(local_input),
            ]
            
            process = await asyncio.create_subprocess_exec(
                *duration_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            duration = float(stdout.decode().strip())

            # Generate thumbnails at even intervals
            thumbnail_keys = []
            for i in range(count):
                timestamp = (duration / (count + 1)) * (i + 1)
                output_file = temp_dir / f"thumb_{i}.jpg"

                # FFmpeg command to extract frame
                cmd = [
                    "ffmpeg",
                    "-ss", str(timestamp),
                    "-i", str(local_input),
                    "-vframes", "1",
                    "-vf", "scale=1280:720",
                    "-q:v", "2",
                    str(output_file),
                ]

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.communicate()

                if output_file.exists():
                    # Upload to S3
                    s3_key = f"videos/{video_id}/thumbnails/thumb_{i}.jpg"
                    with open(output_file, "rb") as f:
                        await self.storage.upload(
                            data=f.read(),
                            key=s3_key,
                            content_type="image/jpeg",
                        )
                    thumbnail_keys.append(s3_key)

            logger.info(f"Generated {len(thumbnail_keys)} thumbnails for video {video_id}")
            return thumbnail_keys

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def get_encoding_progress(self, job_id: UUID) -> Dict[str, any]:
        """
        Get encoding job progress.
        
        Args:
            job_id: Encoding job ID
        
        Returns:
            Dictionary with job status and progress
        """
        result = await self.db.execute(
            select(EncodingJob).where(EncodingJob.id == job_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            raise VideoServiceError(f"Encoding job not found: {job_id}")

        return {
            "job_id": str(job.id),
            "status": job.status.value,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message,
            "output_paths": job.output_paths,
            "hls_manifest": job.hls_manifest_url,
            "dash_manifest": job.dash_manifest_url,
        }
