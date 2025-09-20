import { Injectable, NotFoundException, ForbiddenException, BadRequestException } from '@nestjs/common';
import { AdRepository } from '../shared/database/repositories/ad.repository';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { Ad, AdType, AdStatus, AdTargeting } from '../shared/database/entities/ad.entity';
import { LoggerService } from '../shared/logger/logger.service';

export interface CreateAdRequest {
  title: string;
  description: string;
  type: AdType;
  mediaUrls: string[];
  clickUrl: string;
  budget: number;
  bidAmount: number;
  targeting: AdTargeting;
  targetingCriteria?: Record<string, any>;
  demographics?: Record<string, any>;
  interests?: string[];
  locations?: string[];
  behaviors?: string[];
  startDate?: Date;
  endDate?: Date;
}

export interface UpdateAdRequest {
  title?: string;
  description?: string;
  mediaUrls?: string[];
  clickUrl?: string;
  budget?: number;
  bidAmount?: number;
  targeting?: AdTargeting;
  targetingCriteria?: Record<string, any>;
  demographics?: Record<string, any>;
  interests?: string[];
  locations?: string[];
  behaviors?: string[];
  startDate?: Date;
  endDate?: Date;
  isActive?: boolean;
}

@Injectable()
export class AdsService {
  constructor(
    private adRepository: AdRepository,
    private userRepository: UserRepository,
    private logger: LoggerService,
  ) {}

  async createAd(userId: string, adData: CreateAdRequest): Promise<Ad> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Check if user has sufficient budget
    if (adData.budget <= 0) {
      throw new BadRequestException('Budget must be greater than 0');
    }

    if (adData.bidAmount <= 0) {
      throw new BadRequestException('Bid amount must be greater than 0');
    }

    const ad = await this.adRepository.create({
      title: adData.title,
      description: adData.description,
      type: adData.type,
      mediaUrls: adData.mediaUrls,
      clickUrl: adData.clickUrl,
      budget: adData.budget,
      bidAmount: adData.bidAmount,
      targeting: adData.targeting,
      targetingCriteria: adData.targetingCriteria,
      demographics: adData.demographics,
      interests: adData.interests || [],
      locations: adData.locations || [],
      behaviors: adData.behaviors || [],
      startDate: adData.startDate,
      endDate: adData.endDate,
      advertiserId: userId,
      status: AdStatus.DRAFT,
    });

    this.logger.logBusiness('ad_created', userId, {
      adId: ad.id,
      title: ad.title,
      type: ad.type,
      budget: ad.budget,
    });

    return ad;
  }

  async getAd(adId: string): Promise<Ad> {
    const ad = await this.adRepository.findById(adId);
    if (!ad) {
      throw new NotFoundException('Ad not found');
    }

    return ad;
  }

  async updateAd(adId: string, userId: string, updateData: UpdateAdRequest): Promise<Ad> {
    const ad = await this.adRepository.findById(adId);
    if (!ad) {
      throw new NotFoundException('Ad not found');
    }

    if (ad.advertiserId !== userId) {
      throw new ForbiddenException('Not authorized to update this ad');
    }

    const updatedAd = await this.adRepository.update(adId, updateData);

    this.logger.logBusiness('ad_updated', userId, {
      adId,
      updates: updateData,
    });

    return updatedAd;
  }

  async deleteAd(adId: string, userId: string): Promise<void> {
    const ad = await this.adRepository.findById(adId);
    if (!ad) {
      throw new NotFoundException('Ad not found');
    }

    if (ad.advertiserId !== userId) {
      throw new ForbiddenException('Not authorized to delete this ad');
    }

    await this.adRepository.delete(adId);

    this.logger.logBusiness('ad_deleted', userId, {
      adId,
      title: ad.title,
    });
  }

  async approveAd(adId: string, userId: string): Promise<Ad> {
    const ad = await this.adRepository.findById(adId);
    if (!ad) {
      throw new NotFoundException('Ad not found');
    }

    // Check if user is admin or moderator
    const user = await this.userRepository.findById(userId);
    if (!user || (!user.isAdmin && !user.isModerator)) {
      throw new ForbiddenException('Not authorized to approve ads');
    }

    const updatedAd = await this.adRepository.update(adId, {
      status: AdStatus.APPROVED,
      isActive: true,
    });

    this.logger.logBusiness('ad_approved', userId, {
      adId,
      advertiserId: ad.advertiserId,
    });

    return updatedAd;
  }

  async rejectAd(adId: string, userId: string, reason?: string): Promise<Ad> {
    const ad = await this.adRepository.findById(adId);
    if (!ad) {
      throw new NotFoundException('Ad not found');
    }

    // Check if user is admin or moderator
    const user = await this.userRepository.findById(userId);
    if (!user || (!user.isAdmin && !user.isModerator)) {
      throw new ForbiddenException('Not authorized to reject ads');
    }

    const updatedAd = await this.adRepository.update(adId, {
      status: AdStatus.REJECTED,
      isActive: false,
      metadata: { ...ad.metadata, rejectionReason: reason, rejectedAt: new Date() },
    });

    this.logger.logBusiness('ad_rejected', userId, {
      adId,
      advertiserId: ad.advertiserId,
      reason,
    });

    return updatedAd;
  }

  async getActiveAds(limit: number = 10, offset: number = 0): Promise<Ad[]> {
    return this.adRepository.findActiveAds(limit, offset);
  }

  async getAdsByAdvertiser(advertiserId: string, limit: number = 10, offset: number = 0): Promise<Ad[]> {
    return this.adRepository.findByAdvertiser(advertiserId, limit, offset);
  }

  async getAdsByType(type: AdType, limit: number = 10, offset: number = 0): Promise<Ad[]> {
    return this.adRepository.findByType(type, limit, offset);
  }

  async recordImpression(adId: string): Promise<void> {
    await this.adRepository.incrementImpressions(adId);
    await this.adRepository.updatePerformanceMetrics(adId);

    this.logger.logBusiness('ad_impression', undefined, {
      adId,
    });
  }

  async recordClick(adId: string): Promise<void> {
    await this.adRepository.incrementClicks(adId);
    await this.adRepository.updatePerformanceMetrics(adId);

    this.logger.logBusiness('ad_click', undefined, {
      adId,
    });
  }

  async recordConversion(adId: string): Promise<void> {
    await this.adRepository.incrementConversions(adId);
    await this.adRepository.updatePerformanceMetrics(adId);

    this.logger.logBusiness('ad_conversion', undefined, {
      adId,
    });
  }

  async getAdStats(adId: string): Promise<{
    impressions: number;
    clicks: number;
    conversions: number;
    ctr: number;
    cpm: number;
    cpc: number;
    cpa: number;
    spent: number;
    remainingBudget: number;
    budgetUtilization: number;
    performanceScore: number;
  }> {
    const ad = await this.adRepository.findById(adId);
    if (!ad) {
      throw new NotFoundException('Ad not found');
    }

    return {
      impressions: ad.impressions,
      clicks: ad.clicks,
      conversions: ad.conversions,
      ctr: ad.ctr,
      cpm: ad.cpm,
      cpc: ad.cpc,
      cpa: ad.cpa,
      spent: ad.spent,
      remainingBudget: ad.remainingBudget,
      budgetUtilization: ad.budgetUtilization,
      performanceScore: ad.performanceScore,
    };
  }

  async updateAdStats(adId: string, stats: {
    impressions?: number;
    clicks?: number;
    conversions?: number;
    spent?: number;
  }): Promise<void> {
    await this.adRepository.update(adId, stats);
  }

  async pauseAd(adId: string, userId: string): Promise<Ad> {
    const ad = await this.adRepository.findById(adId);
    if (!ad) {
      throw new NotFoundException('Ad not found');
    }

    if (ad.advertiserId !== userId) {
      throw new ForbiddenException('Not authorized to pause this ad');
    }

    const updatedAd = await this.adRepository.update(adId, {
      isActive: false,
      status: AdStatus.PAUSED,
    });

    this.logger.logBusiness('ad_paused', userId, {
      adId,
    });

    return updatedAd;
  }

  async resumeAd(adId: string, userId: string): Promise<Ad> {
    const ad = await this.adRepository.findById(adId);
    if (!ad) {
      throw new NotFoundException('Ad not found');
    }

    if (ad.advertiserId !== userId) {
      throw new ForbiddenException('Not authorized to resume this ad');
    }

    const updatedAd = await this.adRepository.update(adId, {
      isActive: true,
      status: AdStatus.ACTIVE,
    });

    this.logger.logBusiness('ad_resumed', userId, {
      adId,
    });

    return updatedAd;
  }
}
