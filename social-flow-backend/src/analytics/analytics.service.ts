import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { AnalyticsRepository } from '../shared/database/repositories/analytics.repository';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { Analytics, AnalyticsType, AnalyticsCategory } from '../shared/database/entities/analytics.entity';
import { RedisService } from '../shared/redis/redis.service';
import { LoggerService } from '../shared/logger/logger.service';

export interface TrackEventRequest {
  type: AnalyticsType;
  category: AnalyticsCategory;
  event: string;
  properties?: Record<string, any>;
  context?: Record<string, any>;
  entityType?: string;
  entityId?: string;
}

export interface GetAnalyticsRequest {
  startDate: Date;
  endDate: Date;
  groupBy?: string;
  filters?: Record<string, any>;
}

@Injectable()
export class AnalyticsService {
  constructor(
    private analyticsRepository: AnalyticsRepository,
    private userRepository: UserRepository,
    private redisService: RedisService,
    private logger: LoggerService,
  ) {}

  async trackEvent(userId: string, eventData: TrackEventRequest): Promise<void> {
    try {
      // Add to analytics queue for processing
      await this.redisService.addAnalyticsJob({
        type: eventData.type,
        category: eventData.category,
        userId,
        entityType: eventData.entityType,
        entityId: eventData.entityId,
        event: eventData.event,
        properties: eventData.properties,
        context: eventData.context,
      });

      this.logger.logBusiness('event_tracked', userId, {
        type: eventData.type,
        category: eventData.category,
        event: eventData.event,
        entityType: eventData.entityType,
        entityId: eventData.entityId,
      });
    } catch (error) {
      this.logger.logError(error, 'AnalyticsService.trackEvent', {
        userId,
        eventData,
      });
      throw error;
    }
  }

  async trackPageView(userId: string, page: string, properties?: Record<string, any>): Promise<void> {
    await this.trackEvent(userId, {
      type: AnalyticsType.PAGE_VIEW,
      category: AnalyticsCategory.USER,
      event: 'page_view',
      properties: {
        page,
        ...properties,
      },
    });
  }

  async trackVideoView(userId: string, videoId: string, properties?: Record<string, any>): Promise<void> {
    await this.trackEvent(userId, {
      type: AnalyticsType.VIDEO_VIEW,
      category: AnalyticsCategory.CONTENT,
      event: 'video_view',
      entityType: 'video',
      entityId: videoId,
      properties: {
        videoId,
        ...properties,
      },
    });
  }

  async trackPostView(userId: string, postId: string, properties?: Record<string, any>): Promise<void> {
    await this.trackEvent(userId, {
      type: AnalyticsType.POST_VIEW,
      category: AnalyticsCategory.CONTENT,
      event: 'post_view',
      entityType: 'post',
      entityId: postId,
      properties: {
        postId,
        ...properties,
      },
    });
  }

  async trackUserAction(userId: string, action: string, properties?: Record<string, any>): Promise<void> {
    await this.trackEvent(userId, {
      type: AnalyticsType.USER_ACTION,
      category: AnalyticsCategory.USER,
      event: action,
      properties,
    });
  }

  async trackSystemEvent(event: string, properties?: Record<string, any>): Promise<void> {
    await this.trackEvent(undefined, {
      type: AnalyticsType.SYSTEM_EVENT,
      category: AnalyticsCategory.TECHNICAL,
      event,
      properties,
    });
  }

  async trackPerformance(operation: string, duration: number, properties?: Record<string, any>): Promise<void> {
    await this.trackEvent(undefined, {
      type: AnalyticsType.PERFORMANCE,
      category: AnalyticsCategory.TECHNICAL,
      event: 'performance',
      properties: {
        operation,
        duration,
        ...properties,
      },
    });
  }

  async trackError(error: string, properties?: Record<string, any>): Promise<void> {
    await this.trackEvent(undefined, {
      type: AnalyticsType.ERROR,
      category: AnalyticsCategory.TECHNICAL,
      event: 'error',
      properties: {
        error,
        ...properties,
      },
    });
  }

  async getUserAnalytics(userId: string, request: GetAnalyticsRequest): Promise<any[]> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    return this.analyticsRepository.findByUser(userId, 1000, 0);
  }

  async getContentAnalytics(entityType: string, entityId: string, request: GetAnalyticsRequest): Promise<any[]> {
    return this.analyticsRepository.findByEntity(entityType, entityId, 1000, 0);
  }

  async getSystemAnalytics(request: GetAnalyticsRequest): Promise<any[]> {
    return this.analyticsRepository.findByDateRange(request.startDate, request.endDate, 1000, 0);
  }

  async getAggregatedMetrics(request: GetAnalyticsRequest): Promise<Record<string, any>[]> {
    return this.analyticsRepository.getAggregatedMetrics(
      request.startDate,
      request.endDate,
      request.groupBy || 'date',
    );
  }

  async getTopEvents(limit: number = 10): Promise<Record<string, any>[]> {
    return this.analyticsRepository.getTopEvents(limit);
  }

  async getTopUsers(limit: number = 10): Promise<Record<string, any>[]> {
    return this.analyticsRepository.getTopUsers(limit);
  }

  async getTopEntities(entityType: string, limit: number = 10): Promise<Record<string, any>[]> {
    return this.analyticsRepository.getTopEntities(entityType, limit);
  }

  async getAnalyticsOverview(): Promise<{
    totalEvents: number;
    totalUsers: number;
    topEvents: Record<string, any>[];
    topUsers: Record<string, any>[];
  }> {
    const topEvents = await this.getTopEvents(10);
    const topUsers = await this.getTopUsers(10);

    return {
      totalEvents: 0, // Would need to implement count method
      totalUsers: 0, // Would need to implement count method
      topEvents,
      topUsers,
    };
  }

  async getVideoAnalytics(videoId: string, request: GetAnalyticsRequest): Promise<{
    views: number;
    likes: number;
    comments: number;
    shares: number;
    watchTime: number;
    averageWatchTime: number;
    retentionRate: number;
    demographics: Record<string, any>;
    devices: Record<string, any>;
    sources: Record<string, any>;
  }> {
    // Get video analytics from database
    const analytics = await this.analyticsRepository.findByEntity('video', videoId, 1000, 0);
    
    // Calculate metrics
    const views = analytics.filter(a => a.event === 'video_view').length;
    const likes = analytics.filter(a => a.event === 'video_like').length;
    const comments = analytics.filter(a => a.event === 'video_comment').length;
    const shares = analytics.filter(a => a.event === 'video_share').length;
    const watchTime = analytics
      .filter(a => a.event === 'video_watch_time')
      .reduce((sum, a) => sum + (a.properties?.duration || 0), 0);
    const averageWatchTime = views > 0 ? watchTime / views : 0;
    const retentionRate = analytics
      .filter(a => a.event === 'video_retention')
      .reduce((sum, a) => sum + (a.properties?.rate || 0), 0) / views || 0;

    // Get demographics, devices, and sources
    const demographics = this.getDemographics(analytics);
    const devices = this.getDevices(analytics);
    const sources = this.getSources(analytics);

    return {
      views,
      likes,
      comments,
      shares,
      watchTime,
      averageWatchTime,
      retentionRate,
      demographics,
      devices,
      sources,
    };
  }

  async getUserAnalyticsOverview(userId: string): Promise<{
    totalViews: number;
    totalLikes: number;
    totalComments: number;
    totalShares: number;
    totalPosts: number;
    totalVideos: number;
    engagementRate: number;
    topContent: Record<string, any>[];
  }> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    return {
      totalViews: user.totalViews,
      totalLikes: user.totalLikes,
      totalComments: 0, // Would need to calculate from comments
      totalShares: 0, // Would need to calculate from shares
      totalPosts: user.postsCount,
      totalVideos: user.videosCount,
      engagementRate: 0, // Would need to calculate
      topContent: [], // Would need to implement
    };
  }

  async getContentAnalyticsOverview(): Promise<{
    totalVideos: number;
    totalPosts: number;
    totalViews: number;
    totalLikes: number;
    totalComments: number;
    totalShares: number;
    topVideos: Record<string, any>[];
    topPosts: Record<string, any>[];
  }> {
    // This would need to be implemented with actual data
    return {
      totalVideos: 0,
      totalPosts: 0,
      totalViews: 0,
      totalLikes: 0,
      totalComments: 0,
      totalShares: 0,
      topVideos: [],
      topPosts: [],
    };
  }

  async getRevenueAnalytics(request: GetAnalyticsRequest): Promise<{
    totalRevenue: number;
    revenueBySource: Record<string, number>;
    revenueByPeriod: Record<string, number>;
    topEarners: Record<string, any>[];
  }> {
    // This would need to be implemented with actual payment data
    return {
      totalRevenue: 0,
      revenueBySource: {},
      revenueByPeriod: {},
      topEarners: [],
    };
  }

  async getEngagementAnalytics(request: GetAnalyticsRequest): Promise<{
    totalEngagement: number;
    engagementRate: number;
    engagementByType: Record<string, number>;
    engagementByPeriod: Record<string, number>;
  }> {
    // This would need to be implemented with actual engagement data
    return {
      totalEngagement: 0,
      engagementRate: 0,
      engagementByType: {},
      engagementByPeriod: {},
    };
  }

  async cleanupOldAnalytics(days: number = 90): Promise<void> {
    await this.analyticsRepository.deleteOldAnalytics(days);

    this.logger.logBusiness('old_analytics_cleaned', undefined, {
      days,
    });
  }

  private getDemographics(analytics: Analytics[]): Record<string, any> {
    const demographics = {};
    analytics.forEach(a => {
      if (a.demographics) {
        Object.keys(a.demographics).forEach(key => {
          demographics[key] = (demographics[key] || 0) + 1;
        });
      }
    });
    return demographics;
  }

  private getDevices(analytics: Analytics[]): Record<string, any> {
    const devices = {};
    analytics.forEach(a => {
      if (a.device) {
        devices[a.device] = (devices[a.device] || 0) + 1;
      }
    });
    return devices;
  }

  private getSources(analytics: Analytics[]): Record<string, any> {
    const sources = {};
    analytics.forEach(a => {
      if (a.referrer) {
        const source = this.extractSource(a.referrer);
        sources[source] = (sources[source] || 0) + 1;
      }
    });
    return sources;
  }

  private extractSource(referrer: string): string {
    try {
      const url = new URL(referrer);
      return url.hostname;
    } catch {
      return 'direct';
    }
  }
}
