# Handles video transcoding operations
"""
transcoding.py

High-level FFmpeg orchestration utilities and job management.

Design goals:
- Manage transcode jobs (sync + async)
- Support presets, CRF/bitrate modes, hardware acceleration hooks
- Provide safe subprocess usage and streaming-friendly I/O
- Expose easy-to-consume TranscodeJob objects
"""

import asyncio
import subprocess
import shlex
import uuid
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


@dataclass
class TranscodeJob:
    input_uri: str
    output_uri: str
    vcodec: str = "libx264"
    acodec: str = "aac"
    crf: Optional[int] = 23
    bitrate: Optional[str] = None  # e.g., "1200k"
    preset: str = "medium"
    extra_args: List[str] = field(default_factory=list)
    hw_accel: Optional[str] = None  # e.g., "nvenc", "vaapi"
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class Transcoder:
    """
    Transcoder orchestrates ffmpeg-based transcoding jobs.

    Notes:
    - Uses subprocess for non-blocking, asyncio-aware execution.
    - Builds robust command lines; in production consider templating or a lib wrapper.
    """

    def __init__(self, ffmpeg_cmd: str = "ffmpeg", concurrency: int = 4):
        self.ffmpeg_cmd = ffmpeg_cmd
        self._semaphore = asyncio.Semaphore(concurrency)
        self._jobs: Dict[str, TranscodeJob] = {}
        self.on_progress: Optional[Callable[[str, Dict], None]] = None

    def _build_cmd(self, job: TranscodeJob) -> List[str]:
        parts = [self.ffmpeg_cmd, "-y", "-hide_banner", "-loglevel", "info", "-i", job.input_uri]

        # hardware acceleration hooks (example patterns)
        if job.hw_accel == "nvenc":
            parts += ["-c:v", "h264_nvenc"]
        elif job.hw_accel == "vaapi":
            # VAAPI requires specific devices and format; keep as placeholder
            parts += ["-hwaccel", "vaapi", "-vaapi_device", "/dev/dri/renderD128", "-c:v", "h264_vaapi"]
        else:
            parts += ["-c:v", job.vcodec]

        if job.bitrate:
            parts += ["-b:v", job.bitrate]
        elif job.crf is not None:
            parts += ["-crf", str(job.crf), "-preset", job.preset]

        # audio codec
        parts += ["-c:a", job.acodec]

        # append any extra args supplied
        if job.extra_args:
            parts += job.extra_args

        parts += [job.output_uri]
        return parts

    async def transcode_async(self, job: TranscodeJob, timeout: Optional[int] = None) -> Dict:
        """
        Run ffmpeg job asynchronously, streaming parsing of progress lines.

        Returns dict: {job_id, returncode, stdout, stderr}
        """
        self._jobs[job.job_id] = job
        cmd = self._build_cmd(job)
        # Use shlex for safe logging, but pass list to subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async with self._semaphore:
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                result = {"job_id": job.job_id, "returncode": proc.returncode, "stdout": stdout.decode(errors="ignore"), "stderr": stderr.decode(errors="ignore")}
                return result
            finally:
                # cleanup reference
                self._jobs.pop(job.job_id, None)

    def transcode_blocking(self, job: TranscodeJob, timeout: Optional[int] = None) -> Dict:
        """
        Blocking wrapper for synchronous contexts.
        """
        cmd = self._build_cmd(job)
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
            return {"job_id": job.job_id, "returncode": proc.returncode, "stdout": proc.stdout.decode(errors="ignore"), "stderr": proc.stderr.decode(errors="ignore")}
        except subprocess.TimeoutExpired as e:
            return {"job_id": job.job_id, "returncode": -1, "stdout": "", "stderr": f"Timeout: {e}"}

    def list_jobs(self) -> List[str]:
        return list(self._jobs.keys())

    # Helper presets
    def make_hls_job(self, input_uri: str, output_dir: str, variants: Dict[str, Dict]) -> List[TranscodeJob]:
        """
        Create multiple jobs for HLS renditions.
        `variants` is mapping name -> {resolution, bitrate, crf, vcodec}
        """
        os.makedirs(output_dir, exist_ok=True)
        jobs: List[TranscodeJob] = []
        for name, conf in variants.items():
            out = os.path.join(output_dir, f"{name}.m3u8")
            job = TranscodeJob(
                input_uri=input_uri,
                output_uri=out,
                vcodec=conf.get("vcodec", "libx264"),
                bitrate=conf.get("bitrate"),
                crf=conf.get("crf"),
                preset=conf.get("preset", "fast"),
                extra_args=[
                    "-f", "hls", "-hls_time", str(conf.get("hls_time", 6)), "-hls_playlist_type", "vod",
                    "-hls_segment_filename", os.path.join(output_dir, f"{name}_%03d.ts")
                ],
            )
            jobs.append(job)
        return jobs
