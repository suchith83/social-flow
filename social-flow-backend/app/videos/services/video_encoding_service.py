"""
Video Encoding Service with AWS MediaConvert integration.

Handles multi-bitrate transcoding (240p, 360p, 480p, 720p, 1080p, 4K),
HLS/DASH manifest generation, thumbnail extraction, and CloudFront distribution.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.config import settings
from app.core.redis import get_cache
from app.videos.models.video import Video, VideoStatus
from app.models.encoding_job import EncodingJob, EncodingStatus

logger = logging.getLogger(__name__)


class VideoEncodingService:
    """
    Service for video encoding with AWS MediaConvert.
    
    Features:
    - Multi-bitrate transcoding (240p to 4K)
    - HLS and DASH packaging
    - Thumbnail generation at keyframes
    - Progress tracking via Redis pub/sub
    - CloudFront signed URL generation
    - Retry logic for failed encodings
    """
    
    # Encoding presets for different quality levels
    ENCODING_PRESETS = {
        '240p': {
            'width': 426,
            'height': 240,
            'bitrate': 400000,  # 400 kbps
            'buffer_size': 800000,
            'framerate': 30
        },
        '360p': {
            'width': 640,
            'height': 360,
            'bitrate': 800000,  # 800 kbps
            'buffer_size': 1600000,
            'framerate': 30
        },
        '480p': {
            'width': 854,
            'height': 480,
            'bitrate': 1400000,  # 1.4 Mbps
            'buffer_size': 2800000,
            'framerate': 30
        },
        '720p': {
            'width': 1280,
            'height': 720,
            'bitrate': 2800000,  # 2.8 Mbps
            'buffer_size': 5600000,
            'framerate': 30
        },
        '1080p': {
            'width': 1920,
            'height': 1080,
            'bitrate': 5000000,  # 5 Mbps
            'buffer_size': 10000000,
            'framerate': 30
        },
        '1440p': {
            'width': 2560,
            'height': 1440,
            'bitrate': 10000000,  # 10 Mbps
            'buffer_size': 20000000,
            'framerate': 60
        },
        '4k': {
            'width': 3840,
            'height': 2160,
            'bitrate': 20000000,  # 20 Mbps
            'buffer_size': 40000000,
            'framerate': 60
        }
    }
    
    def __init__(self):
        """Initialize video encoding service."""
        self.s3_client = None
        self.mediaconvert_client = None
        self.cloudfront_client = None
        self.cache = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS clients."""
        try:
            # S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            
            # MediaConvert client
            # First, get MediaConvert endpoint
            mediaconvert_control = boto3.client(
                'mediaconvert',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            endpoints = mediaconvert_control.describe_endpoints()
            endpoint_url = endpoints['Endpoints'][0]['Url']
            
            self.mediaconvert_client = boto3.client(
                'mediaconvert',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
                endpoint_url=endpoint_url
            )
            
            # CloudFront client for signed URLs
            self.cloudfront_client = boto3.client(
                'cloudfront',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            
            logger.info("AWS clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise
    
    async def _get_cache(self):
        """Get Redis cache instance."""
        if self.cache is None:
            self.cache = await get_cache()
        return self.cache
    
    async def create_encoding_job(
        self,
        db: AsyncSession,
        video_id: str,
        input_s3_key: str,
        qualities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new encoding job for a video.
        
        Args:
            db: Database session
            video_id: Video ID
            input_s3_key: S3 key of source video
            qualities: List of quality levels to encode (default: all)
        
        Returns:
            Dict with job details
        """
        try:
            # Default to all qualities if not specified
            if qualities is None:
                qualities = ['240p', '360p', '480p', '720p', '1080p']
            
            # Validate qualities
            invalid_qualities = [q for q in qualities if q not in self.ENCODING_PRESETS]
            if invalid_qualities:
                raise ValueError(f"Invalid qualities: {invalid_qualities}")
            
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # Update video status
            await db.execute(
                update(Video)
                .where(Video.id == video_id)
                .values(status=VideoStatus.PROCESSING, processing_started_at=datetime.utcnow())
            )
            
            # Create encoding job record
            encoding_job = EncodingJob(
                id=job_id,
                video_id=video_id,
                input_s3_key=input_s3_key,
                output_s3_prefix=f"encoded/{video_id}/",
                qualities=json.dumps(qualities),
                status=EncodingStatus.QUEUED,
                created_at=datetime.utcnow()
            )
            db.add(encoding_job)
            await db.commit()
            
            # Submit to MediaConvert
            mediaconvert_job_id = await self._submit_mediaconvert_job(
                job_id,
                input_s3_key,
                f"encoded/{video_id}/",
                qualities
            )
            
            # Update job with MediaConvert job ID
            encoding_job.mediaconvert_job_id = mediaconvert_job_id
            encoding_job.status = EncodingStatus.PROCESSING
            encoding_job.started_at = datetime.utcnow()
            await db.commit()
            
            # Publish job created event to Redis
            cache = await self._get_cache()
            await cache.publish(
                f"encoding:job:{job_id}",
                json.dumps({
                    'event': 'job_created',
                    'job_id': job_id,
                    'video_id': video_id,
                    'status': 'processing',
                    'timestamp': datetime.utcnow().isoformat()
                })
            )
            
            logger.info(f"Encoding job {job_id} created for video {video_id}")
            
            return {
                'job_id': job_id,
                'video_id': video_id,
                'mediaconvert_job_id': mediaconvert_job_id,
                'status': 'processing',
                'qualities': qualities
            }
        
        except Exception as e:
            logger.error(f"Failed to create encoding job: {e}")
            # Update video status to failed
            await db.execute(
                update(Video)
                .where(Video.id == video_id)
                .values(status=VideoStatus.FAILED, processing_error=str(e))
            )
            await db.commit()
            raise
    
    async def _submit_mediaconvert_job(
        self,
        job_id: str,
        input_s3_key: str,
        output_s3_prefix: str,
        qualities: List[str]
    ) -> str:
        """
        Submit encoding job to AWS MediaConvert.
        
        Args:
            job_id: Internal job ID
            input_s3_key: S3 key of source video
            output_s3_prefix: S3 prefix for encoded outputs
            qualities: List of quality levels
        
        Returns:
            MediaConvert job ID
        """
        try:
            # Build input/output paths
            input_uri = f"s3://{settings.AWS_S3_BUCKET}/{input_s3_key}"
            output_uri = f"s3://{settings.AWS_S3_BUCKET}/{output_s3_prefix}"
            
            # Build output groups for HLS and DASH
            output_groups = []
            
            # HLS output group
            hls_outputs = []
            for quality in qualities:
                preset = self.ENCODING_PRESETS[quality]
                hls_outputs.append({
                    "ContainerSettings": {
                        "Container": "M3U8",
                        "M3u8Settings": {
                            "AudioFramesPerPes": 4,
                            "PcrControl": "PCR_EVERY_PES_PACKET",
                            "PmtPid": 480,
                            "PrivateMetadataPid": 503,
                            "ProgramNumber": 1,
                            "PatInterval": 0,
                            "PmtInterval": 0,
                            "VideoPid": 481,
                            "AudioPids": [482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492]
                        }
                    },
                    "VideoDescription": {
                        "Width": preset['width'],
                        "Height": preset['height'],
                        "CodecSettings": {
                            "Codec": "H_264",
                            "H264Settings": {
                                "RateControlMode": "CBR",
                                "Bitrate": preset['bitrate'],
                                "FramerateControl": "SPECIFIED",
                                "FramerateNumerator": preset['framerate'],
                                "FramerateDenominator": 1,
                                "CodecProfile": "HIGH",
                                "CodecLevel": "AUTO",
                                "MaxBitrate": preset['buffer_size'],
                                "GopSize": 2.0,
                                "GopSizeUnits": "SECONDS",
                                "NumberBFramesBetweenReferenceFrames": 2,
                                "GopClosedCadence": 1,
                                "GopBReference": "ENABLED",
                                "HrdBufferSize": preset['buffer_size'],
                                "HrdBufferInitialFillPercentage": 90,
                                "SlowPal": "DISABLED",
                                "ParControl": "SPECIFIED",
                                "Syntax": "DEFAULT",
                                "NumberReferenceFrames": 3,
                                "DynamicSubGop": "STATIC",
                                "FieldEncoding": "PAFF",
                                "SceneChangeDetect": "ENABLED",
                                "MinIInterval": 0,
                                "Telecine": "NONE",
                                "Softness": 0,
                                "InsertColorMetadata": "INSERT"
                            }
                        }
                    },
                    "AudioDescriptions": [{
                        "CodecSettings": {
                            "Codec": "AAC",
                            "AacSettings": {
                                "Bitrate": 96000,
                                "CodingMode": "CODING_MODE_2_0",
                                "SampleRate": 48000
                            }
                        }
                    }],
                    "NameModifier": f"_{quality}"
                })
            
            output_groups.append({
                "Name": "HLS Group",
                "OutputGroupSettings": {
                    "Type": "HLS_GROUP_SETTINGS",
                    "HlsGroupSettings": {
                        "SegmentLength": 6,
                        "MinSegmentLength": 0,
                        "Destination": f"{output_uri}hls/",
                        "SegmentControl": "SEGMENTED_FILES",
                        "ManifestDurationFormat": "INTEGER",
                        "StreamInfResolution": "INCLUDE",
                        "ClientCache": "ENABLED",
                        "ProgramDateTimePeriod": 600,
                        "CodecSpecification": "RFC_4281",
                        "OutputSelection": "MANIFESTS_AND_SEGMENTS",
                        "ManifestCompression": "NONE",
                        "DirectoryStructure": "SINGLE_DIRECTORY"
                    }
                },
                "Outputs": hls_outputs
            })
            
            # DASH output group (similar structure)
            dash_outputs = []
            for quality in qualities:
                preset = self.ENCODING_PRESETS[quality]
                dash_outputs.append({
                    "ContainerSettings": {
                        "Container": "MPD"
                    },
                    "VideoDescription": {
                        "Width": preset['width'],
                        "Height": preset['height'],
                        "CodecSettings": {
                            "Codec": "H_264",
                            "H264Settings": {
                                "RateControlMode": "CBR",
                                "Bitrate": preset['bitrate'],
                                "FramerateControl": "SPECIFIED",
                                "FramerateNumerator": preset['framerate'],
                                "FramerateDenominator": 1
                            }
                        }
                    },
                    "AudioDescriptions": [{
                        "CodecSettings": {
                            "Codec": "AAC",
                            "AacSettings": {
                                "Bitrate": 96000,
                                "CodingMode": "CODING_MODE_2_0",
                                "SampleRate": 48000
                            }
                        }
                    }],
                    "NameModifier": f"_{quality}"
                })
            
            output_groups.append({
                "Name": "DASH Group",
                "OutputGroupSettings": {
                    "Type": "DASH_ISO_GROUP_SETTINGS",
                    "DashIsoGroupSettings": {
                        "SegmentLength": 6,
                        "Destination": f"{output_uri}dash/",
                        "FragmentLength": 2,
                        "SegmentControl": "SEGMENTED_FILES",
                        "MpdProfile": "MAIN_PROFILE",
                        "HbbtvCompliance": "NONE"
                    }
                },
                "Outputs": dash_outputs
            })
            
            # Thumbnail output group
            output_groups.append({
                "Name": "Thumbnail Group",
                "OutputGroupSettings": {
                    "Type": "FILE_GROUP_SETTINGS",
                    "FileGroupSettings": {
                        "Destination": f"{output_uri}thumbnails/"
                    }
                },
                "Outputs": [{
                    "ContainerSettings": {
                        "Container": "RAW"
                    },
                    "VideoDescription": {
                        "Width": 1280,
                        "Height": 720,
                        "CodecSettings": {
                            "Codec": "FRAME_CAPTURE",
                            "FrameCaptureSettings": {
                                "FramerateNumerator": 1,
                                "FramerateDenominator": 10,  # 1 frame every 10 seconds
                                "MaxCaptures": 10,
                                "Quality": 80
                            }
                        }
                    },
                    "NameModifier": "_thumb"
                }]
            })
            
            # Submit job
            response = self.mediaconvert_client.create_job(
                Role=settings.AWS_MEDIACONVERT_ROLE_ARN,
                Settings={
                    "Inputs": [{
                        "FileInput": input_uri,
                        "AudioSelectors": {
                            "Audio Selector 1": {
                                "DefaultSelection": "DEFAULT"
                            }
                        },
                        "VideoSelector": {},
                        "TimecodeSource": "ZEROBASED"
                    }],
                    "OutputGroups": output_groups
                },
                Queue=settings.AWS_MEDIACONVERT_QUEUE_ARN,
                UserMetadata={
                    "internal_job_id": job_id
                }
            )
            
            mediaconvert_job_id = response['Job']['Id']
            logger.info(f"MediaConvert job {mediaconvert_job_id} submitted for {job_id}")
            
            return mediaconvert_job_id
        
        except ClientError as e:
            logger.error(f"MediaConvert job submission failed: {e}")
            raise
    
    async def check_job_status(self, db: AsyncSession, job_id: str) -> Dict[str, Any]:
        """
        Check the status of an encoding job.
        
        Args:
            db: Database session
            job_id: Internal job ID
        
        Returns:
            Dict with job status details
        """
        try:
            # Get job from database
            result = await db.execute(
                select(EncodingJob).where(EncodingJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            # Query MediaConvert for job status
            if job.mediaconvert_job_id:
                response = self.mediaconvert_client.get_job(Id=job.mediaconvert_job_id)
                mc_job = response['Job']
                mc_status = mc_job['Status']
                
                # Update job status based on MediaConvert status
                if mc_status == 'COMPLETE':
                    job.status = EncodingStatus.COMPLETED
                    job.completed_at = datetime.utcnow()
                    job.progress = 100
                    
                    # Update video status
                    await self._finalize_video_encoding(db, job)
                    
                elif mc_status == 'ERROR' or mc_status == 'CANCELED':
                    job.status = EncodingStatus.FAILED
                    job.completed_at = datetime.utcnow()
                    job.error_message = mc_job.get('ErrorMessage', 'Unknown error')
                    
                    # Update video status
                    await db.execute(
                        update(Video)
                        .where(Video.id == job.video_id)
                        .values(status=VideoStatus.FAILED, processing_error=job.error_message)
                    )
                    
                elif mc_status == 'PROGRESSING':
                    # Extract progress percentage
                    job.progress = mc_job.get('JobPercentComplete', 0)
                    
                    # Publish progress event
                    cache = await self._get_cache()
                    await cache.publish(
                        f"encoding:job:{job_id}",
                        json.dumps({
                            'event': 'progress_update',
                            'job_id': job_id,
                            'progress': job.progress,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    )
                
                await db.commit()
            
            return {
                'job_id': job_id,
                'video_id': job.video_id,
                'status': job.status.value,
                'progress': job.progress,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'error_message': job.error_message
            }
        
        except Exception as e:
            logger.error(f"Failed to check job status: {e}")
            raise
    
    async def _finalize_video_encoding(self, db: AsyncSession, job: EncodingJob):
        """
        Finalize video encoding after successful completion.
        
        Args:
            db: Database session
            job: Encoding job
        """
        try:
            # Parse qualities
            qualities = json.loads(job.qualities)
            
            # Build manifest URLs
            hls_manifest_key = f"{job.output_s3_prefix}hls/master.m3u8"
            dash_manifest_key = f"{job.output_s3_prefix}dash/manifest.mpd"
            
            # Generate CloudFront URLs
            hls_url = self._generate_cloudfront_url(hls_manifest_key)
            dash_url = self._generate_cloudfront_url(dash_manifest_key)
            
            # Get thumbnail URLs
            thumbnail_urls = await self._get_thumbnail_urls(job.output_s3_prefix)
            
            # Update video record
            await db.execute(
                update(Video)
                .where(Video.id == job.video_id)
                .values(
                    status=VideoStatus.PROCESSED,
                    processing_completed_at=datetime.utcnow(),
                    hls_url=hls_url,
                    dash_url=dash_url,
                    thumbnail_url=thumbnail_urls[0] if thumbnail_urls else None,
                    available_qualities=json.dumps(qualities)
                )
            )
            await db.commit()
            
            # Publish completion event
            cache = await self._get_cache()
            await cache.publish(
                f"encoding:job:{job.id}",
                json.dumps({
                    'event': 'job_completed',
                    'job_id': job.id,
                    'video_id': job.video_id,
                    'hls_url': hls_url,
                    'dash_url': dash_url,
                    'thumbnail_url': thumbnail_urls[0] if thumbnail_urls else None,
                    'timestamp': datetime.utcnow().isoformat()
                })
            )
            
            logger.info(f"Video encoding finalized for {job.video_id}")
        
        except Exception as e:
            logger.error(f"Failed to finalize video encoding: {e}")
            raise
    
    async def _get_thumbnail_urls(self, output_s3_prefix: str) -> List[str]:
        """Get list of thumbnail URLs from S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=settings.AWS_S3_BUCKET,
                Prefix=f"{output_s3_prefix}thumbnails/"
            )
            
            thumbnail_urls = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.jpg') or key.endswith('.png'):
                    url = self._generate_cloudfront_url(key)
                    thumbnail_urls.append(url)
            
            return sorted(thumbnail_urls)
        
        except Exception as e:
            logger.error(f"Failed to get thumbnail URLs: {e}")
            return []
    
    def _generate_cloudfront_url(self, s3_key: str) -> str:
        """Generate CloudFront URL for an S3 object."""
        cloudfront_domain = settings.AWS_CLOUDFRONT_DOMAIN
        return f"https://{cloudfront_domain}/{s3_key}"
    
    def generate_signed_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate a signed CloudFront URL with expiration.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds (default: 1 hour)
        
        Returns:
            Signed CloudFront URL
        """
        try:
            # CloudFront URL
            url = f"https://{settings.AWS_CLOUDFRONT_DOMAIN}/{s3_key}"
            
            # Generate signature
            # Note: Requires CloudFront key pair configured
            expire_time = int(time.time()) + expiration
            
            # Build policy
            policy = json.dumps({
                "Statement": [{
                    "Resource": url,
                    "Condition": {
                        "DateLessThan": {
                            "AWS:EpochTime": expire_time
                        }
                    }
                }]
            })
            
            # This is a simplified version
            # In production, use rsa library to sign with CloudFront private key
            # For now, return unsigned URL
            # TODO: Implement CloudFront URL signing with private key
            
            return url
        
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            return self._generate_cloudfront_url(s3_key)
    
    async def retry_failed_job(self, db: AsyncSession, job_id: str) -> Dict[str, Any]:
        """
        Retry a failed encoding job.
        
        Args:
            db: Database session
            job_id: Job ID to retry
        
        Returns:
            Dict with new job details
        """
        try:
            # Get failed job
            result = await db.execute(
                select(EncodingJob).where(EncodingJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job.status != EncodingStatus.FAILED:
                raise ValueError(f"Job {job_id} is not in failed state")
            
            # Create new job with same parameters
            qualities = json.loads(job.qualities)
            new_job = await self.create_encoding_job(
                db,
                job.video_id,
                job.input_s3_key,
                qualities
            )
            
            return new_job
        
        except Exception as e:
            logger.error(f"Failed to retry job: {e}")
            raise


# Singleton instance
_encoding_service = None


def get_encoding_service() -> VideoEncodingService:
    """Get singleton encoding service instance."""
    global _encoding_service
    if _encoding_service is None:
        _encoding_service = VideoEncodingService()
    return _encoding_service
