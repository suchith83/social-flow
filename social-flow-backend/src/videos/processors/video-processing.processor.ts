import { Process, Processor } from '@nestjs/bull';
import { Job } from 'bull';
import { LoggerService } from '../../shared/logger/logger.service';
import { VideoProcessingService } from '../video-processing.service';
import { VideosService } from '../videos.service';

@Processor('video-processing')
export class VideoProcessingProcessor {
  constructor(
    private logger: LoggerService,
    private videoProcessingService: VideoProcessingService,
    private videosService: VideosService,
  ) {}

  @Process('process-video')
  async handleVideoProcessing(job: Job<any>) {
    const { videoId, filePath, userId, metadata } = job.data;

    try {
      this.logger.logBusiness('video_processing_started', userId, {
        videoId,
        filePath,
        jobId: job.id,
      });

      // Update video status to processing
      await this.videosService.updateVideoProcessingStatus(videoId, 'processing');

      // Create transcoding job
      const transcodingJobId = await this.videoProcessingService.createTranscodingJob(
        videoId,
        filePath,
      );

      // Wait for transcoding to complete (in production, you'd use webhooks)
      // For now, we'll simulate completion
      await this.simulateTranscodingCompletion(videoId, transcodingJobId);

      // Update video status to processed
      await this.videosService.updateVideoProcessingStatus(videoId, 'processed', {
        transcodingJobId,
        completedAt: new Date(),
      });

      this.logger.logBusiness('video_processing_completed', userId, {
        videoId,
        transcodingJobId,
        jobId: job.id,
      });
    } catch (error) {
      this.logger.logError(error, 'VideoProcessingProcessor.handleVideoProcessing', {
        videoId,
        userId,
        jobId: job.id,
      });

      // Update video status to failed
      await this.videosService.updateVideoProcessingStatus(videoId, 'failed', {
        error: error.message,
        failedAt: new Date(),
      });

      throw error;
    }
  }

  private async simulateTranscodingCompletion(videoId: string, transcodingJobId: string) {
    // In production, this would be handled by AWS MediaConvert webhooks
    // For now, we'll simulate a delay
    await new Promise(resolve => setTimeout(resolve, 5000));

    // Simulate updating video URLs
    await this.videosService.updateVideoUrls(videoId, {
      hlsUrl: `https://cdn.socialflow.com/videos/${videoId}/hls/playlist.m3u8`,
      dashUrl: `https://cdn.socialflow.com/videos/${videoId}/dash/manifest.mpd`,
      previewUrl: `https://cdn.socialflow.com/videos/${videoId}/preview.mp4`,
      resolutions: {
        '240p': `https://cdn.socialflow.com/videos/${videoId}/240p.mp4`,
        '360p': `https://cdn.socialflow.com/videos/${videoId}/360p.mp4`,
        '480p': `https://cdn.socialflow.com/videos/${videoId}/480p.mp4`,
        '720p': `https://cdn.socialflow.com/videos/${videoId}/720p.mp4`,
        '1080p': `https://cdn.socialflow.com/videos/${videoId}/1080p.mp4`,
      },
      thumbnails: {
        default: `https://cdn.socialflow.com/videos/${videoId}/thumbnails/default.jpg`,
        medium: `https://cdn.socialflow.com/videos/${videoId}/thumbnails/medium.jpg`,
        high: `https://cdn.socialflow.com/videos/${videoId}/thumbnails/high.jpg`,
      },
    });
  }
}
