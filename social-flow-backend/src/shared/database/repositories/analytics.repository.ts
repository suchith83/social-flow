import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Analytics, AnalyticsType, AnalyticsCategory } from '../entities/analytics.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class AnalyticsRepository extends BaseRepository<Analytics> {
  constructor(
    @InjectRepository(Analytics)
    private readonly analyticsRepository: Repository<Analytics>,
  ) {
    super(analyticsRepository);
  }

  async findByUser(userId: string, limit: number = 10, offset: number = 0): Promise<Analytics[]> {
    return this.analyticsRepository.find({
      where: { userId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByType(type: AnalyticsType, limit: number = 10, offset: number = 0): Promise<Analytics[]> {
    return this.analyticsRepository.find({
      where: { type },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByCategory(category: AnalyticsCategory, limit: number = 10, offset: number = 0): Promise<Analytics[]> {
    return this.analyticsRepository.find({
      where: { category },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByEntity(entityType: string, entityId: string, limit: number = 10, offset: number = 0): Promise<Analytics[]> {
    return this.analyticsRepository.find({
      where: { entityType, entityId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByDateRange(startDate: Date, endDate: Date, limit: number = 10, offset: number = 0): Promise<Analytics[]> {
    return this.analyticsRepository.find({
      where: {
        createdAt: {
          $gte: startDate,
          $lte: endDate,
        } as any,
      },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async getAggregatedMetrics(
    startDate: Date,
    endDate: Date,
    groupBy: string = 'date'
  ): Promise<Record<string, any>[]> {
    return this.analyticsRepository
      .createQueryBuilder('analytics')
      .select(`${groupBy}(analytics.createdAt)`, 'period')
      .addSelect('COUNT(*)', 'count')
      .addSelect('SUM(analytics.value)', 'totalValue')
      .addSelect('AVG(analytics.value)', 'avgValue')
      .where('analytics.createdAt BETWEEN :startDate AND :endDate', { startDate, endDate })
      .groupBy(`${groupBy}(analytics.createdAt)`)
      .orderBy('period', 'ASC')
      .getRawMany();
  }

  async getTopEvents(limit: number = 10): Promise<Record<string, any>[]> {
    return this.analyticsRepository
      .createQueryBuilder('analytics')
      .select('analytics.event', 'event')
      .addSelect('COUNT(*)', 'count')
      .groupBy('analytics.event')
      .orderBy('count', 'DESC')
      .take(limit)
      .getRawMany();
  }

  async getTopUsers(limit: number = 10): Promise<Record<string, any>[]> {
    return this.analyticsRepository
      .createQueryBuilder('analytics')
      .select('analytics.userId', 'userId')
      .addSelect('COUNT(*)', 'count')
      .where('analytics.userId IS NOT NULL')
      .groupBy('analytics.userId')
      .orderBy('count', 'DESC')
      .take(limit)
      .getRawMany();
  }

  async getTopEntities(entityType: string, limit: number = 10): Promise<Record<string, any>[]> {
    return this.analyticsRepository
      .createQueryBuilder('analytics')
      .select('analytics.entityId', 'entityId')
      .addSelect('COUNT(*)', 'count')
      .where('analytics.entityType = :entityType', { entityType })
      .groupBy('analytics.entityId')
      .orderBy('count', 'DESC')
      .take(limit)
      .getRawMany();
  }

  async deleteOldAnalytics(days: number = 90): Promise<void> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - days);

    await this.analyticsRepository
      .createQueryBuilder()
      .delete()
      .where('createdAt < :cutoffDate', { cutoffDate })
      .execute();
  }
}
