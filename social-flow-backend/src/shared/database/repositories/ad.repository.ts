import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Ad, AdStatus, AdType } from '../entities/ad.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class AdRepository extends BaseRepository<Ad> {
  constructor(
    @InjectRepository(Ad)
    private readonly adRepository: Repository<Ad>,
  ) {
    super(adRepository);
  }

  async findActiveAds(limit: number = 10, offset: number = 0): Promise<Ad[]> {
    return this.adRepository.find({
      where: { status: AdStatus.ACTIVE, isActive: true },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByAdvertiser(advertiserId: string, limit: number = 10, offset: number = 0): Promise<Ad[]> {
    return this.adRepository.find({
      where: { advertiserId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByType(type: AdType, limit: number = 10, offset: number = 0): Promise<Ad[]> {
    return this.adRepository.find({
      where: { type, status: AdStatus.ACTIVE },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async incrementImpressions(adId: string): Promise<void> {
    await this.adRepository.increment({ id: adId }, 'impressions', 1);
  }

  async incrementClicks(adId: string): Promise<void> {
    await this.adRepository.increment({ id: adId }, 'clicks', 1);
  }

  async incrementConversions(adId: string): Promise<void> {
    await this.adRepository.increment({ id: adId }, 'conversions', 1);
  }

  async updatePerformanceMetrics(adId: string): Promise<void> {
    await this.adRepository
      .createQueryBuilder()
      .update(Ad)
      .set({
        ctr: () => 'CASE WHEN impressions > 0 THEN (clicks::float / impressions) * 100 ELSE 0 END',
        cpm: () => 'CASE WHEN impressions > 0 THEN (spent::float / impressions) * 1000 ELSE 0 END',
        cpc: () => 'CASE WHEN clicks > 0 THEN spent::float / clicks ELSE 0 END',
        cpa: () => 'CASE WHEN conversions > 0 THEN spent::float / conversions ELSE 0 END',
      })
      .where('id = :id', { id: adId })
      .execute();
  }
}
