import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { ViewCount } from '../entities/view-count.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class ViewCountRepository extends BaseRepository<ViewCount> {
  constructor(
    @InjectRepository(ViewCount)
    private readonly viewCountRepository: Repository<ViewCount>,
  ) {
    super(viewCountRepository);
  }

  async findByVideo(videoId: string, limit: number = 10, offset: number = 0): Promise<ViewCount[]> {
    return this.viewCountRepository.find({
      where: { videoId },
      take: limit,
      skip: offset,
      order: { date: 'DESC' },
    });
  }

  async findByDate(date: Date): Promise<ViewCount[]> {
    return this.viewCountRepository.find({
      where: { date },
      order: { count: 'DESC' },
    });
  }

  async findByVideoAndDate(videoId: string, date: Date): Promise<ViewCount | null> {
    return this.viewCountRepository.findOne({ where: { videoId, date } });
  }

  async incrementViews(videoId: string, date: Date, views: number = 1): Promise<void> {
    const viewCount = await this.findByVideoAndDate(videoId, date);
    if (viewCount) {
      await this.viewCountRepository.increment({ id: viewCount.id }, 'count', views);
    } else {
      await this.create({
        videoId,
        date,
        count: views,
        uniqueViews: 1,
        watchTime: 0,
        averageWatchTime: 0,
        retentionRate: 0,
      });
    }
  }

  async incrementUniqueViews(videoId: string, date: Date): Promise<void> {
    const viewCount = await this.findByVideoAndDate(videoId, date);
    if (viewCount) {
      await this.viewCountRepository.increment({ id: viewCount.id }, 'uniqueViews', 1);
    } else {
      await this.create({
        videoId,
        date,
        count: 1,
        uniqueViews: 1,
        watchTime: 0,
        averageWatchTime: 0,
        retentionRate: 0,
      });
    }
  }

  async updateWatchTime(videoId: string, date: Date, watchTime: number): Promise<void> {
    const viewCount = await this.findByVideoAndDate(videoId, date);
    if (viewCount) {
      await this.viewCountRepository
        .createQueryBuilder()
        .update(ViewCount)
        .set({
          watchTime: () => `watch_time + ${watchTime}`,
          averageWatchTime: () => `watch_time / GREATEST(unique_views, 1)`,
        })
        .where('id = :id', { id: viewCount.id })
        .execute();
    }
  }

  async getTotalViews(videoId: string): Promise<number> {
    const result = await this.viewCountRepository
      .createQueryBuilder('viewCount')
      .select('SUM(viewCount.count)', 'total')
      .where('viewCount.videoId = :videoId', { videoId })
      .getRawOne();
    return parseInt(result.total) || 0;
  }

  async getTotalUniqueViews(videoId: string): Promise<number> {
    const result = await this.viewCountRepository
      .createQueryBuilder('viewCount')
      .select('SUM(viewCount.uniqueViews)', 'total')
      .where('viewCount.videoId = :videoId', { videoId })
      .getRawOne();
    return parseInt(result.total) || 0;
  }

  async getTotalWatchTime(videoId: string): Promise<number> {
    const result = await this.viewCountRepository
      .createQueryBuilder('viewCount')
      .select('SUM(viewCount.watchTime)', 'total')
      .where('viewCount.videoId = :videoId', { videoId })
      .getRawOne();
    return parseInt(result.total) || 0;
  }

  async getTopVideos(limit: number = 10, startDate?: Date, endDate?: Date): Promise<Record<string, any>[]> {
    const query = this.viewCountRepository
      .createQueryBuilder('viewCount')
      .select('viewCount.videoId', 'videoId')
      .addSelect('SUM(viewCount.count)', 'totalViews')
      .addSelect('SUM(viewCount.uniqueViews)', 'totalUniqueViews')
      .addSelect('SUM(viewCount.watchTime)', 'totalWatchTime')
      .groupBy('viewCount.videoId')
      .orderBy('totalViews', 'DESC')
      .take(limit);

    if (startDate) {
      query.andWhere('viewCount.date >= :startDate', { startDate });
    }

    if (endDate) {
      query.andWhere('viewCount.date <= :endDate', { endDate });
    }

    return query.getRawMany();
  }

  async getViewsByDateRange(
    videoId: string,
    startDate: Date,
    endDate: Date
  ): Promise<ViewCount[]> {
    return this.viewCountRepository
      .createQueryBuilder('viewCount')
      .where('viewCount.videoId = :videoId', { videoId })
      .andWhere('viewCount.date BETWEEN :startDate AND :endDate', { startDate, endDate })
      .orderBy('viewCount.date', 'ASC')
      .getMany();
  }

  async deleteOldViewCounts(days: number = 365): Promise<void> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - days);

    await this.viewCountRepository
      .createQueryBuilder()
      .delete()
      .where('date < :cutoffDate', { cutoffDate })
      .execute();
  }
}
