import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Video, VideoStatus, VideoVisibility } from '../entities/video.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class VideoRepository extends BaseRepository<Video> {
  constructor(
    @InjectRepository(Video)
    private readonly videoRepository: Repository<Video>,
  ) {
    super(videoRepository);
  }

  async findByUserId(userId: string, limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository.find({
      where: { userId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findPublicVideos(limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository.find({
      where: { 
        visibility: VideoVisibility.PUBLIC,
        status: VideoStatus.PROCESSED,
      },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findTrendingVideos(limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository
      .createQueryBuilder('video')
      .where('video.visibility = :visibility', { visibility: VideoVisibility.PUBLIC })
      .andWhere('video.status = :status', { status: VideoStatus.PROCESSED })
      .orderBy('video.viewsCount', 'DESC')
      .addOrderBy('video.likesCount', 'DESC')
      .take(limit)
      .skip(offset)
      .getMany();
  }

  async findVideosByTag(tag: string, limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository
      .createQueryBuilder('video')
      .where('video.visibility = :visibility', { visibility: VideoVisibility.PUBLIC })
      .andWhere('video.status = :status', { status: VideoStatus.PROCESSED })
      .andWhere(':tag = ANY(video.tags)', { tag })
      .orderBy('video.createdAt', 'DESC')
      .take(limit)
      .skip(offset)
      .getMany();
  }

  async searchVideos(query: string, limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository
      .createQueryBuilder('video')
      .where('video.visibility = :visibility', { visibility: VideoVisibility.PUBLIC })
      .andWhere('video.status = :status', { status: VideoStatus.PROCESSED })
      .andWhere('(video.title ILIKE :query OR video.description ILIKE :query)', {
        query: `%${query}%`,
      })
      .orderBy('video.createdAt', 'DESC')
      .take(limit)
      .skip(offset)
      .getMany();
  }

  async findVideosByStatus(status: VideoStatus, limit: number = 10, offset: number = 0): Promise<Video[]> {
    return this.videoRepository.find({
      where: { status },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findLiveVideos(): Promise<Video[]> {
    return this.videoRepository.find({
      where: { isLive: true },
      order: { createdAt: 'DESC' },
    });
  }

  async incrementViewsCount(videoId: string): Promise<void> {
    await this.videoRepository.increment({ id: videoId }, 'viewsCount', 1);
  }

  async incrementLikesCount(videoId: string): Promise<void> {
    await this.videoRepository.increment({ id: videoId }, 'likesCount', 1);
  }

  async decrementLikesCount(videoId: string): Promise<void> {
    await this.videoRepository.decrement({ id: videoId }, 'likesCount', 1);
  }

  async incrementCommentsCount(videoId: string): Promise<void> {
    await this.videoRepository.increment({ id: videoId }, 'commentsCount', 1);
  }

  async decrementCommentsCount(videoId: string): Promise<void> {
    await this.videoRepository.decrement({ id: videoId }, 'commentsCount', 1);
  }

  async incrementSharesCount(videoId: string): Promise<void> {
    await this.videoRepository.increment({ id: videoId }, 'sharesCount', 1);
  }

  async updateWatchTime(videoId: string, watchTime: number): Promise<void> {
    await this.videoRepository
      .createQueryBuilder()
      .update(Video)
      .set({
        watchTime: () => `watch_time + ${watchTime}`,
        averageWatchTime: () => `watch_time / GREATEST(views_count, 1)`,
        retentionRate: () => `(watch_time / GREATEST(duration, 1)) * 100`,
      })
      .where('id = :id', { id: videoId })
      .execute();
  }

  async updateProcessingStatus(videoId: string, status: VideoStatus, metadata?: Record<string, any>): Promise<void> {
    const updateData: Partial<Video> = { status };
    if (metadata) {
      updateData.processingMetadata = metadata;
    }
    await this.videoRepository.update(videoId, updateData);
  }

  async updateVideoUrls(videoId: string, urls: {
    videoUrl?: string;
    hlsUrl?: string;
    dashUrl?: string;
    previewUrl?: string;
    resolutions?: Record<string, string>;
    thumbnails?: Record<string, string>;
  }): Promise<void> {
    await this.videoRepository.update(videoId, urls);
  }

  async findVideosNeedingProcessing(): Promise<Video[]> {
    return this.videoRepository.find({
      where: { status: VideoStatus.UPLOADING },
      order: { createdAt: 'ASC' },
    });
  }

  async findVideosByDateRange(startDate: Date, endDate: Date): Promise<Video[]> {
    return this.videoRepository
      .createQueryBuilder('video')
      .where('video.createdAt BETWEEN :startDate AND :endDate', { startDate, endDate })
      .orderBy('video.createdAt', 'DESC')
      .getMany();
  }

  async countVideosByUser(userId: string): Promise<number> {
    return this.videoRepository.count({ where: { userId } });
  }

  async countPublicVideos(): Promise<number> {
    return this.videoRepository.count({ 
      where: { 
        visibility: VideoVisibility.PUBLIC,
        status: VideoStatus.PROCESSED,
      },
    });
  }

  async countVideosByStatus(status: VideoStatus): Promise<number> {
    return this.videoRepository.count({ where: { status } });
  }
}
