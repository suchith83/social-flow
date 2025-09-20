import { Injectable, NotFoundException, ForbiddenException, BadRequestException } from '@nestjs/common';
import { VideoRepository } from '../shared/database/repositories/video.repository';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { ViewCountRepository } from '../shared/database/repositories/view-count.repository';
import { Video, VideoStatus, VideoVisibility } from '../shared/database/entities/video.entity';
import { S3Service } from '../shared/aws/s3.service';
import { VideoProcessingService } from './video-processing.service';
import { LoggerService } from '../shared/logger/logger.service';

export interface UploadVideoRequest {
  title: string;
  description?: string;
  tags?: string[];
  visibility?: VideoVisibility;
  isMonetized?: boolean;
  isAgeRestricted?: boolean;
  scheduledAt?: Date;
}

export interface UpdateVideoRequest {
  title?: string;
  description?: string;
  tags?: string[];
  visibility?: VideoVisibility;
  isMonetized?: boolean;
  isAgeRestricted?: boolean;
  scheduledAt?: Date;
}

export interface VideoUploadResponse {
  videoId: string;
  uploadUrl: string;
  video: Video;
}

@Injectable()
export class VideosService {
  constructor(
    private videoRepository: VideoRepository,
    private userRepository: UserRepository,
    private viewCountRepository: ViewCountRepository,
    private s3Service: S3Service,
    private videoProcessingService: VideoProcessingService,
    private logger: LoggerService,
  ) {}

  async uploadVideo(
    userId: string,
    uploadData: UploadVideoRequest,
    file: Express.Multer.File,
  ): Promise<VideoUploadResponse> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Create video record
    const video = await this.videoRepository.create({
      title: uploadData.title,
      description: uploadData.description,
      tags: uploadData.tags || [],
      visibility: uploadData.visibility || VideoVisibility.PUBLIC,
      isMonetized: uploadData.isMonetized || false,
      isAgeRestricted: uploadData.isAgeRestricted || false,
      scheduledAt: uploadData.scheduledAt,
      userId,
      status: VideoStatus.UPLOADING,
      fileSize: file.size,
      width: 0, // Will be updated after processing
      height: 0, // Will be updated after processing
      duration: 0, // Will be updated after processing
    });

    // Upload to S3
    const videoUrl = await this.s3Service.uploadVideo(video.id, file.buffer, file.originalname);

    // Update video with URL
    await this.videoRepository.update(video.id, { videoUrl });

    // Start processing job
    await this.videoProcessingService.processVideo({
      videoId: video.id,
      filePath: videoUrl,
      userId,
      metadata: {
        title: uploadData.title,
        description: uploadData.description,
        tags: uploadData.tags,
        visibility: uploadData.visibility,
        isMonetized: uploadData.isMonetized,
        isAgeRestricted: uploadData.isAgeRestricted,
      },
    });

    // Update user video count
    await this.userRepository.incrementVideosCount(userId);

    this.logger.logBusiness('video_uploaded', userId, {
      videoId: video.id,
      title: uploadData.title,
      fileSize: file.size,
    });

    return {
      videoId: video.id,
      uploadUrl: videoUrl,
      video,
    };
  }

  async getVideo(videoId: string, userId?: string): Promise<Video> {
    const video = await this.videoRepository.findById(videoId);
    if (!video) {
      throw new NotFoundException('Video not found');
    }

    // Check visibility
    if (video.visibility === VideoVisibility.PRIVATE && video.userId !== userId) {
      throw new ForbiddenException('Video is private');
    }

    return video;
  }

  async updateVideo(
    videoId: string,
    userId: string,
    updateData: UpdateVideoRequest,
  ): Promise<Video> {
    const video = await this.videoRepository.findById(videoId);
    if (!video) {
      throw new NotFoundException('Video not found');
    }

    if (video.userId !== userId) {
      throw new ForbiddenException('Not authorized to update this video');
    }

    const updatedVideo = await this.videoRepository.update(videoId, updateData);

    this.logger.logBusiness('video_updated', userId, {
      videoId,
      updates: updateData,
    });

    return updatedVideo;
  }

  async deleteVideo(videoId: string, userId: string): Promise<void> {
    const video = await this.videoRepository.findById(videoId);
    if (!video) {
      throw new NotFoundException('Video not found');
    }

    if (video.userId !== userId) {
      throw new ForbiddenException('Not authorized to delete this video');
    }

    // Delete from S3
    if (video.videoUrl) {
      await this.s3Service.deleteFile(video.videoUrl);
    }

    // Delete video record
    await this.videoRepository.delete(videoId);

    // Update user video count
    await this.userRepository.decrementVideosCount(userId);

    this.logger.logBusiness('video_deleted', userId, {
      videoId,
      title: video.title,
    });
  }

  async getVideoStream(videoId: string, userId?: string): Promise<string> {
    const video = await this.getVideo(videoId, userId);
    
    if (!video.isProcessed) {
      throw new BadRequestException('Video is still processing');
    }

    // Return HLS URL if available, otherwise return video URL
    return video.hlsUrl || video.videoUrl;
  }

  async getVideoThumbnail(videoId: string, size: string = 'default'): Promise<string> {
    const video = await this.videoRepository.findById(videoId);
    if (!video) {
      throw new NotFoundException('Video not found');
    }

    // Return thumbnail URL from S3
    return this.s3Service.getThumbnailUrl(videoId, size);
  }

  async recordView(videoId: string, userId?: string): Promise<void> {
    const video = await this.videoRepository.findById(videoId);
    if (!video) {
      throw new NotFoundException('Video not found');
    }

    // Increment view count
    await this.videoRepository.incrementViewsCount(videoId);

    // Update user total views
    if (video.userId) {
      await this.userRepository.incrementTotalViews(video.userId);
    }

    // Record view analytics
    const today = new Date();
    await this.viewCountRepository.incrementViews(videoId, today);

    this.logger.logBusiness('video_viewed', userId, {
      videoId,
      videoTitle: video.title,
      videoUserId: video.userId,
    });
  }

  async getTrendingVideos(limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository.findTrendingVideos(limit, offset);
  }

  async getVideosByTag(tag: string, limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository.findVideosByTag(tag, limit, offset);
  }

  async searchVideos(query: string, limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository.searchVideos(query, limit, offset);
  }

  async getUserVideos(userId: string, limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository.findByUserId(userId, limit, offset);
  }

  async getPublicVideos(limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository.findPublicVideos(limit, offset);
  }

  async getVideoStats(videoId: string): Promise<{
    viewsCount: number;
    likesCount: number;
    commentsCount: number;
    sharesCount: number;
    watchTime: number;
    averageWatchTime: number;
    retentionRate: number;
  }> {
    const video = await this.videoRepository.findById(videoId);
    if (!video) {
      throw new NotFoundException('Video not found');
    }

    return {
      viewsCount: video.viewsCount,
      likesCount: video.likesCount,
      commentsCount: video.commentsCount,
      sharesCount: video.sharesCount,
      watchTime: video.watchTime,
      averageWatchTime: video.averageWatchTime,
      retentionRate: video.retentionRate,
    };
  }

  async updateVideoStats(videoId: string, stats: {
    viewsCount?: number;
    likesCount?: number;
    commentsCount?: number;
    sharesCount?: number;
    watchTime?: number;
    averageWatchTime?: number;
    retentionRate?: number;
  }): Promise<void> {
    await this.videoRepository.update(videoId, stats);
  }

  async getVideosNeedingProcessing(): Promise<Video[]> {
    return this.videoRepository.findVideosNeedingProcessing();
  }

  async updateVideoProcessingStatus(
    videoId: string,
    status: VideoStatus,
    metadata?: Record<string, any>,
  ): Promise<void> {
    await this.videoRepository.updateProcessingStatus(videoId, status, metadata);
  }

  async updateVideoUrls(
    videoId: string,
    urls: {
      videoUrl?: string;
      hlsUrl?: string;
      dashUrl?: string;
      previewUrl?: string;
      resolutions?: Record<string, string>;
      thumbnails?: Record<string, string>;
    },
  ): Promise<void> {
    await this.videoRepository.updateVideoUrls(videoId, urls);
  }
}
