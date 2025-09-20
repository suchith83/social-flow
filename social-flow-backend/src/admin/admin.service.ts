import { Injectable, NotFoundException, ForbiddenException, BadRequestException } from '@nestjs/common';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { VideoRepository } from '../shared/database/repositories/video.repository';
import { PostRepository } from '../shared/database/repositories/post.repository';
import { AdRepository } from '../shared/database/repositories/ad.repository';
import { PaymentRepository } from '../shared/database/repositories/payment.repository';
import { SubscriptionRepository } from '../shared/database/repositories/subscription.repository';
import { NotificationRepository } from '../shared/database/repositories/notification.repository';
import { AnalyticsRepository } from '../shared/database/repositories/analytics.repository';
import { ViewCountRepository } from '../shared/database/repositories/view-count.repository';
import { User } from '../shared/database/entities/user.entity';
import { Video } from '../shared/database/entities/video.entity';
import { Post } from '../shared/database/entities/post.entity';
import { Ad } from '../shared/database/entities/ad.entity';
import { Payment } from '../shared/database/entities/payment.entity';
import { Subscription } from '../shared/database/entities/subscription.entity';
import { Notification } from '../shared/database/entities/notification.entity';
import { Analytics } from '../shared/database/entities/analytics.entity';
import { ViewCount } from '../shared/database/entities/view-count.entity';
import { LoggerService } from '../shared/logger/logger.service';

export interface AdminStats {
  totalUsers: number;
  totalVideos: number;
  totalPosts: number;
  totalAds: number;
  totalRevenue: number;
  totalViews: number;
  totalLikes: number;
  totalComments: number;
  totalShares: number;
  activeUsers: number;
  newUsersToday: number;
  newVideosToday: number;
  newPostsToday: number;
  revenueToday: number;
  viewsToday: number;
  likesToday: number;
  commentsToday: number;
  sharesToday: number;
}

export interface UserManagementRequest {
  userId: string;
  action: 'ban' | 'unban' | 'suspend' | 'unsuspend' | 'delete';
  reason?: string;
  duration?: number; // in days for suspension
}

export interface ContentModerationRequest {
  entityType: 'video' | 'post' | 'comment';
  entityId: string;
  action: 'approve' | 'reject' | 'flag' | 'unflag';
  reason?: string;
  moderatorId: string;
}

export interface AdManagementRequest {
  adId: string;
  action: 'approve' | 'reject' | 'pause' | 'resume';
  reason?: string;
  moderatorId: string;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    database: 'up' | 'down';
    redis: 'up' | 'down';
    aws: 'up' | 'down';
    storage: 'up' | 'down';
  };
  metrics: {
    responseTime: number;
    errorRate: number;
    throughput: number;
    memoryUsage: number;
    cpuUsage: number;
  };
  alerts: string[];
}

@Injectable()
export class AdminService {
  constructor(
    private userRepository: UserRepository,
    private videoRepository: VideoRepository,
    private postRepository: PostRepository,
    private adRepository: AdRepository,
    private paymentRepository: PaymentRepository,
    private subscriptionRepository: SubscriptionRepository,
    private notificationRepository: NotificationRepository,
    private analyticsRepository: AnalyticsRepository,
    private viewCountRepository: ViewCountRepository,
    private logger: LoggerService,
  ) {}

  async getAdminStats(): Promise<AdminStats> {
    try {
      const [
        totalUsers,
        totalVideos,
        totalPosts,
        totalAds,
        totalRevenue,
        totalViews,
        totalLikes,
        totalComments,
        totalShares,
        activeUsers,
        newUsersToday,
        newVideosToday,
        newPostsToday,
        revenueToday,
        viewsToday,
        likesToday,
        commentsToday,
        sharesToday,
      ] = await Promise.all([
        this.userRepository.count(),
        this.videoRepository.count(),
        this.postRepository.count(),
        this.adRepository.count(),
        this.getTotalRevenue(),
        this.getTotalViews(),
        this.getTotalLikes(),
        this.getTotalComments(),
        this.getTotalShares(),
        this.getActiveUsers(),
        this.getNewUsersToday(),
        this.getNewVideosToday(),
        this.getNewPostsToday(),
        this.getRevenueToday(),
        this.getViewsToday(),
        this.getLikesToday(),
        this.getCommentsToday(),
        this.getSharesToday(),
      ]);

      return {
        totalUsers,
        totalVideos,
        totalPosts,
        totalAds,
        totalRevenue,
        totalViews,
        totalLikes,
        totalComments,
        totalShares,
        activeUsers,
        newUsersToday,
        newVideosToday,
        newPostsToday,
        revenueToday,
        viewsToday,
        likesToday,
        commentsToday,
        sharesToday,
      };
    } catch (error) {
      this.logger.logError(error, 'AdminService.getAdminStats');
      throw error;
    }
  }

  async manageUser(request: UserManagementRequest): Promise<void> {
    const user = await this.userRepository.findById(request.userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    switch (request.action) {
      case 'ban':
        await this.banUser(user, request.reason);
        break;
      case 'unban':
        await this.unbanUser(user);
        break;
      case 'suspend':
        await this.suspendUser(user, request.duration, request.reason);
        break;
      case 'unsuspend':
        await this.unsuspendUser(user);
        break;
      case 'delete':
        await this.deleteUser(user);
        break;
      default:
        throw new BadRequestException('Invalid action');
    }

    this.logger.logBusiness('user_managed', request.userId, {
      action: request.action,
      reason: request.reason,
      duration: request.duration,
    });
  }

  async moderateContent(request: ContentModerationRequest): Promise<void> {
    let entity;
    switch (request.entityType) {
      case 'video':
        entity = await this.videoRepository.findById(request.entityId);
        break;
      case 'post':
        entity = await this.postRepository.findById(request.entityId);
        break;
      case 'comment':
        entity = await this.postRepository.findById(request.entityId); // Assuming comment is a post
        break;
      default:
        throw new BadRequestException('Invalid entity type');
    }

    if (!entity) {
      throw new NotFoundException('Entity not found');
    }

    switch (request.action) {
      case 'approve':
        await this.approveContent(entity, request.moderatorId);
        break;
      case 'reject':
        await this.rejectContent(entity, request.reason, request.moderatorId);
        break;
      case 'flag':
        await this.flagContent(entity, request.reason, request.moderatorId);
        break;
      case 'unflag':
        await this.unflagContent(entity, request.moderatorId);
        break;
      default:
        throw new BadRequestException('Invalid action');
    }

    this.logger.logBusiness('content_moderated', request.entityId, {
      entityType: request.entityType,
      action: request.action,
      reason: request.reason,
      moderatorId: request.moderatorId,
    });
  }

  async manageAd(request: AdManagementRequest): Promise<void> {
    const ad = await this.adRepository.findById(request.adId);
    if (!ad) {
      throw new NotFoundException('Ad not found');
    }

    switch (request.action) {
      case 'approve':
        await this.approveAd(ad, request.moderatorId);
        break;
      case 'reject':
        await this.rejectAd(ad, request.reason, request.moderatorId);
        break;
      case 'pause':
        await this.pauseAd(ad, request.moderatorId);
        break;
      case 'resume':
        await this.resumeAd(ad, request.moderatorId);
        break;
      default:
        throw new BadRequestException('Invalid action');
    }

    this.logger.logBusiness('ad_managed', request.adId, {
      action: request.action,
      reason: request.reason,
      moderatorId: request.moderatorId,
    });
  }

  async getSystemHealth(): Promise<SystemHealth> {
    try {
      const services = await this.checkServices();
      const metrics = await this.getSystemMetrics();
      const alerts = await this.getSystemAlerts();

      let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
      if (Object.values(services).some(s => s === 'down')) {
        status = 'unhealthy';
      } else if (metrics.errorRate > 0.05 || metrics.responseTime > 1000) {
        status = 'degraded';
      }

      return {
        status,
        services,
        metrics,
        alerts,
      };
    } catch (error) {
      this.logger.logError(error, 'AdminService.getSystemHealth');
      throw error;
    }
  }

  async getUsers(page: number = 1, limit: number = 20): Promise<{ users: User[]; total: number }> {
    const offset = (page - 1) * limit;
    const users = await this.userRepository.findAll(limit, offset);
    const total = await this.userRepository.count();
    return { users, total };
  }

  async getVideos(page: number = 1, limit: number = 20): Promise<{ videos: Video[]; total: number }> {
    const offset = (page - 1) * limit;
    const videos = await this.videoRepository.findAll(limit, offset);
    const total = await this.videoRepository.count();
    return { videos, total };
  }

  async getPosts(page: number = 1, limit: number = 20): Promise<{ posts: Post[]; total: number }> {
    const offset = (page - 1) * limit;
    const posts = await this.postRepository.findAll(limit, offset);
    const total = await this.postRepository.count();
    return { posts, total };
  }

  async getAds(page: number = 1, limit: number = 20): Promise<{ ads: Ad[]; total: number }> {
    const offset = (page - 1) * limit;
    const ads = await this.adRepository.findAll(limit, offset);
    const total = await this.adRepository.count();
    return { ads, total };
  }

  async getPayments(page: number = 1, limit: number = 20): Promise<{ payments: Payment[]; total: number }> {
    const offset = (page - 1) * limit;
    const payments = await this.paymentRepository.findAll(limit, offset);
    const total = await this.paymentRepository.count();
    return { payments, total };
  }

  async getSubscriptions(page: number = 1, limit: number = 20): Promise<{ subscriptions: Subscription[]; total: number }> {
    const offset = (page - 1) * limit;
    const subscriptions = await this.subscriptionRepository.findAll(limit, offset);
    const total = await this.subscriptionRepository.count();
    return { subscriptions, total };
  }

  async getNotifications(page: number = 1, limit: number = 20): Promise<{ notifications: Notification[]; total: number }> {
    const offset = (page - 1) * limit;
    const notifications = await this.notificationRepository.findAll(limit, offset);
    const total = await this.notificationRepository.count();
    return { notifications, total };
  }

  async getAnalytics(page: number = 1, limit: number = 20): Promise<{ analytics: Analytics[]; total: number }> {
    const offset = (page - 1) * limit;
    const analytics = await this.analyticsRepository.findAll(limit, offset);
    const total = await this.analyticsRepository.count();
    return { analytics, total };
  }

  async getViewCounts(page: number = 1, limit: number = 20): Promise<{ viewCounts: ViewCount[]; total: number }> {
    const offset = (page - 1) * limit;
    const viewCounts = await this.viewCountRepository.findAll(limit, offset);
    const total = await this.viewCountRepository.count();
    return { viewCounts, total };
  }

  private async banUser(user: User, reason?: string): Promise<void> {
    user.isBanned = true;
    user.banReason = reason;
    user.bannedAt = new Date();
    await this.userRepository.update(user.id, user);
  }

  private async unbanUser(user: User): Promise<void> {
    user.isBanned = false;
    user.banReason = null;
    user.bannedAt = null;
    await this.userRepository.update(user.id, user);
  }

  private async suspendUser(user: User, duration?: number, reason?: string): Promise<void> {
    user.isSuspended = true;
    user.suspensionReason = reason;
    user.suspendedAt = new Date();
    if (duration) {
      user.suspensionEndsAt = new Date(Date.now() + duration * 24 * 60 * 60 * 1000);
    }
    await this.userRepository.update(user.id, user);
  }

  private async unsuspendUser(user: User): Promise<void> {
    user.isSuspended = false;
    user.suspensionReason = null;
    user.suspendedAt = null;
    user.suspensionEndsAt = null;
    await this.userRepository.update(user.id, user);
  }

  private async deleteUser(user: User): Promise<void> {
    await this.userRepository.delete(user.id);
  }

  private async approveContent(entity: any, moderatorId: string): Promise<void> {
    entity.isApproved = true;
    entity.approvedAt = new Date();
    entity.approvedBy = moderatorId;
    await this.updateEntity(entity);
  }

  private async rejectContent(entity: any, reason?: string, moderatorId?: string): Promise<void> {
    entity.isRejected = true;
    entity.rejectedAt = new Date();
    entity.rejectionReason = reason;
    entity.rejectedBy = moderatorId;
    await this.updateEntity(entity);
  }

  private async flagContent(entity: any, reason?: string, moderatorId?: string): Promise<void> {
    entity.isFlagged = true;
    entity.flaggedAt = new Date();
    entity.flagReason = reason;
    entity.flaggedBy = moderatorId;
    await this.updateEntity(entity);
  }

  private async unflagContent(entity: any, moderatorId?: string): Promise<void> {
    entity.isFlagged = false;
    entity.flaggedAt = null;
    entity.flagReason = null;
    entity.flaggedBy = null;
    await this.updateEntity(entity);
  }

  private async updateEntity(entity: any): Promise<void> {
    if (entity instanceof Video) {
      await this.videoRepository.update(entity.id, entity);
    } else if (entity instanceof Post) {
      await this.postRepository.update(entity.id, entity);
    }
  }

  private async approveAd(ad: Ad, moderatorId: string): Promise<void> {
    ad.isApproved = true;
    ad.approvedAt = new Date();
    ad.approvedBy = moderatorId;
    await this.adRepository.update(ad.id, ad);
  }

  private async rejectAd(ad: Ad, reason?: string, moderatorId?: string): Promise<void> {
    ad.isRejected = true;
    ad.rejectedAt = new Date();
    ad.rejectionReason = reason;
    ad.rejectedBy = moderatorId;
    await this.adRepository.update(ad.id, ad);
  }

  private async pauseAd(ad: Ad, moderatorId?: string): Promise<void> {
    ad.isPaused = true;
    ad.pausedAt = new Date();
    ad.pausedBy = moderatorId;
    await this.adRepository.update(ad.id, ad);
  }

  private async resumeAd(ad: Ad, moderatorId?: string): Promise<void> {
    ad.isPaused = false;
    ad.pausedAt = null;
    ad.pausedBy = null;
    await this.adRepository.update(ad.id, ad);
  }

  private async checkServices(): Promise<{
    database: 'up' | 'down';
    redis: 'up' | 'down';
    aws: 'up' | 'down';
    storage: 'up' | 'down';
  }> {
    // This would need to be implemented with actual health checks
    return {
      database: 'up',
      redis: 'up',
      aws: 'up',
      storage: 'up',
    };
  }

  private async getSystemMetrics(): Promise<{
    responseTime: number;
    errorRate: number;
    throughput: number;
    memoryUsage: number;
    cpuUsage: number;
  }> {
    // This would need to be implemented with actual metrics
    return {
      responseTime: 0,
      errorRate: 0,
      throughput: 0,
      memoryUsage: 0,
      cpuUsage: 0,
    };
  }

  private async getSystemAlerts(): Promise<string[]> {
    // This would need to be implemented with actual alert checking
    return [];
  }

  private async getTotalRevenue(): Promise<number> {
    // This would need to be implemented with actual revenue calculation
    return 0;
  }

  private async getTotalViews(): Promise<number> {
    // This would need to be implemented with actual views calculation
    return 0;
  }

  private async getTotalLikes(): Promise<number> {
    // This would need to be implemented with actual likes calculation
    return 0;
  }

  private async getTotalComments(): Promise<number> {
    // This would need to be implemented with actual comments calculation
    return 0;
  }

  private async getTotalShares(): Promise<number> {
    // This would need to be implemented with actual shares calculation
    return 0;
  }

  private async getActiveUsers(): Promise<number> {
    // This would need to be implemented with actual active users calculation
    return 0;
  }

  private async getNewUsersToday(): Promise<number> {
    // This would need to be implemented with actual new users calculation
    return 0;
  }

  private async getNewVideosToday(): Promise<number> {
    // This would need to be implemented with actual new videos calculation
    return 0;
  }

  private async getNewPostsToday(): Promise<number> {
    // This would need to be implemented with actual new posts calculation
    return 0;
  }

  private async getRevenueToday(): Promise<number> {
    // This would need to be implemented with actual revenue calculation
    return 0;
  }

  private async getViewsToday(): Promise<number> {
    // This would need to be implemented with actual views calculation
    return 0;
  }

  private async getLikesToday(): Promise<number> {
    // This would need to be implemented with actual likes calculation
    return 0;
  }

  private async getCommentsToday(): Promise<number> {
    // This would need to be implemented with actual comments calculation
    return 0;
  }

  private async getSharesToday(): Promise<number> {
    // This would need to be implemented with actual shares calculation
    return 0;
  }
}
