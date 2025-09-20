import { Injectable } from '@nestjs/common';
import { RedisService } from '../shared/redis/redis.service';
import { MediaConvertService } from '../shared/aws/media-convert.service';
import { S3Service } from '../shared/aws/s3.service';
import { LoggerService } from '../shared/logger/logger.service';

export interface VideoProcessingJob {
  videoId: string;
  filePath: string;
  userId: string;
  metadata: Record<string, any>;
}

@Injectable()
export class VideoProcessingService {
  constructor(
    private redisService: RedisService,
    private mediaConvertService: MediaConvertService,
    private s3Service: S3Service,
    private logger: LoggerService,
  ) {}

  async processVideo(job: VideoProcessingJob): Promise<void> {
    try {
      // Add to processing queue
      await this.redisService.addVideoProcessingJob(job);

      this.logger.logBusiness('video_processing_started', job.userId, {
        videoId: job.videoId,
        filePath: job.filePath,
      });
    } catch (error) {
      this.logger.logError(error, 'VideoProcessingService.processVideo', {
        videoId: job.videoId,
        userId: job.userId,
      });
      throw error;
    }
  }

  async generateThumbnails(videoId: string, filePath: string, timestamps: number[]): Promise<void> {
    try {
      await this.redisService.addThumbnailGenerationJob({
        videoId,
        filePath,
        timestamps,
      });

      this.logger.logBusiness('thumbnail_generation_started', undefined, {
        videoId,
        filePath,
        timestamps,
      });
    } catch (error) {
      this.logger.logError(error, 'VideoProcessingService.generateThumbnails', {
        videoId,
        filePath,
      });
      throw error;
    }
  }

  async createTranscodingJob(videoId: string, inputKey: string): Promise<string> {
    try {
      const job = await this.mediaConvertService.createTranscodingJob({
        videoId,
        inputKey,
        outputPrefix: `videos/${videoId}`,
        resolutions: ['240p', '360p', '480p', '720p', '1080p'],
        thumbnails: true,
        metadata: {
          videoId,
        },
      });

      this.logger.logBusiness('transcoding_job_created', undefined, {
        videoId,
        jobId: job,
        inputKey,
      });

      return job;
    } catch (error) {
      this.logger.logError(error, 'VideoProcessingService.createTranscodingJob', {
        videoId,
        inputKey,
      });
      throw error;
    }
  }

  async getTranscodingJobStatus(jobId: string): Promise<any> {
    try {
      return await this.mediaConvertService.getJobStatus(jobId);
    } catch (error) {
      this.logger.logError(error, 'VideoProcessingService.getTranscodingJobStatus', {
        jobId,
      });
      throw error;
    }
  }

  async cancelTranscodingJob(jobId: string): Promise<void> {
    try {
      await this.mediaConvertService.cancelJob(jobId);

      this.logger.logBusiness('transcoding_job_cancelled', undefined, {
        jobId,
      });
    } catch (error) {
      this.logger.logError(error, 'VideoProcessingService.cancelTranscodingJob', {
        jobId,
      });
      throw error;
    }
  }

  async listTranscodingJobs(status?: string, maxResults?: number): Promise<any[]> {
    try {
      return await this.mediaConvertService.listJobs(status, maxResults);
    } catch (error) {
      this.logger.logError(error, 'VideoProcessingService.listTranscodingJobs', {
        status,
        maxResults,
      });
      throw error;
    }
  }
}
