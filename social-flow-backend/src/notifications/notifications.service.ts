import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { NotificationRepository } from '../shared/database/repositories/notification.repository';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { Notification, NotificationType, NotificationStatus } from '../shared/database/entities/notification.entity';
import { EmailService } from './email.service';
import { PushNotificationService } from './push-notification.service';
import { LoggerService } from '../shared/logger/logger.service';

export interface CreateNotificationRequest {
  type: NotificationType;
  title: string;
  message: string;
  imageUrl?: string;
  actionUrl?: string;
  relatedId?: string;
  data?: Record<string, any>;
}

export interface SendNotificationRequest {
  userId: string;
  type: NotificationType;
  title: string;
  message: string;
  imageUrl?: string;
  actionUrl?: string;
  relatedId?: string;
  data?: Record<string, any>;
  sendEmail?: boolean;
  sendPush?: boolean;
}

@Injectable()
export class NotificationsService {
  constructor(
    private notificationRepository: NotificationRepository,
    private userRepository: UserRepository,
    private emailService: EmailService,
    private pushNotificationService: PushNotificationService,
    private logger: LoggerService,
  ) {}

  async createNotification(userId: string, notificationData: CreateNotificationRequest): Promise<Notification> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    const notification = await this.notificationRepository.create({
      type: notificationData.type,
      title: notificationData.title,
      message: notificationData.message,
      imageUrl: notificationData.imageUrl,
      actionUrl: notificationData.actionUrl,
      relatedId: notificationData.relatedId,
      data: notificationData.data,
      userId,
      status: NotificationStatus.UNREAD,
    });

    this.logger.logBusiness('notification_created', userId, {
      notificationId: notification.id,
      type: notificationData.type,
      title: notificationData.title,
    });

    return notification;
  }

  async sendNotification(notificationData: SendNotificationRequest): Promise<Notification> {
    const user = await this.userRepository.findById(notificationData.userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Create notification record
    const notification = await this.createNotification(notificationData.userId, {
      type: notificationData.type,
      title: notificationData.title,
      message: notificationData.message,
      imageUrl: notificationData.imageUrl,
      actionUrl: notificationData.actionUrl,
      relatedId: notificationData.relatedId,
      data: notificationData.data,
    });

    // Send email if requested
    if (notificationData.sendEmail) {
      await this.emailService.sendNotificationEmail(user.email, {
        title: notificationData.title,
        message: notificationData.message,
        actionUrl: notificationData.actionUrl,
        imageUrl: notificationData.imageUrl,
      });
    }

    // Send push notification if requested
    if (notificationData.sendPush) {
      await this.pushNotificationService.sendPushNotification(notificationData.userId, {
        title: notificationData.title,
        body: notificationData.message,
        data: notificationData.data,
        imageUrl: notificationData.imageUrl,
      });
    }

    return notification;
  }

  async getNotification(notificationId: string, userId: string): Promise<Notification> {
    const notification = await this.notificationRepository.findById(notificationId);
    if (!notification) {
      throw new NotFoundException('Notification not found');
    }

    if (notification.userId !== userId) {
      throw new ForbiddenException('Not authorized to view this notification');
    }

    return notification;
  }

  async getUserNotifications(userId: string, limit: number = 10, offset: number = 0): Promise<Notification[]> {
    return this.notificationRepository.findByUser(userId, limit, offset);
  }

  async getUnreadNotifications(userId: string, limit: number = 10, offset: number = 0): Promise<Notification[]> {
    return this.notificationRepository.findUnreadByUser(userId, limit, offset);
  }

  async markAsRead(notificationId: string, userId: string): Promise<void> {
    const notification = await this.getNotification(notificationId, userId);
    await this.notificationRepository.markAsRead(notificationId);

    this.logger.logBusiness('notification_read', userId, {
      notificationId,
    });
  }

  async markAllAsRead(userId: string): Promise<void> {
    await this.notificationRepository.markAllAsRead(userId);

    this.logger.logBusiness('all_notifications_read', userId, {});
  }

  async deleteNotification(notificationId: string, userId: string): Promise<void> {
    const notification = await this.getNotification(notificationId, userId);
    await this.notificationRepository.delete(notificationId);

    this.logger.logBusiness('notification_deleted', userId, {
      notificationId,
    });
  }

  async getUnreadCount(userId: string): Promise<number> {
    return this.notificationRepository.countUnreadByUser(userId);
  }

  async getNotificationStats(userId: string): Promise<{
    total: number;
    unread: number;
    byType: Record<string, number>;
  }> {
    const total = await this.notificationRepository.count({ where: { userId } });
    const unread = await this.notificationRepository.countUnreadByUser(userId);

    // Get counts by type (this would require a more complex query in production)
    const byType = {
      [NotificationType.LIKE]: 0,
      [NotificationType.COMMENT]: 0,
      [NotificationType.FOLLOW]: 0,
      [NotificationType.MENTION]: 0,
      [NotificationType.SHARE]: 0,
      [NotificationType.VIDEO_UPLOADED]: 0,
      [NotificationType.VIDEO_PROCESSED]: 0,
      [NotificationType.PAYMENT_RECEIVED]: 0,
      [NotificationType.PAYMENT_FAILED]: 0,
      [NotificationType.SUBSCRIPTION_CREATED]: 0,
      [NotificationType.SUBSCRIPTION_CANCELLED]: 0,
      [NotificationType.AD_APPROVED]: 0,
      [NotificationType.AD_REJECTED]: 0,
      [NotificationType.SYSTEM_ANNOUNCEMENT]: 0,
      [NotificationType.SECURITY_ALERT]: 0,
    };

    return { total, unread, byType };
  }

  // System notification methods
  async sendSystemAnnouncement(
    title: string,
    message: string,
    actionUrl?: string,
    imageUrl?: string,
    targetUsers?: string[],
  ): Promise<void> {
    const users = targetUsers 
      ? await this.userRepository.findUsersByIds(targetUsers)
      : await this.userRepository.findActiveUsers(1000, 0);

    for (const user of users) {
      await this.sendNotification({
        userId: user.id,
        type: NotificationType.SYSTEM_ANNOUNCEMENT,
        title,
        message,
        actionUrl,
        imageUrl,
        sendEmail: true,
        sendPush: true,
      });
    }

    this.logger.logBusiness('system_announcement_sent', undefined, {
      title,
      targetUsers: targetUsers?.length || users.length,
    });
  }

  async sendSecurityAlert(
    userId: string,
    title: string,
    message: string,
    actionUrl?: string,
  ): Promise<void> {
    await this.sendNotification({
      userId,
      type: NotificationType.SECURITY_ALERT,
      title,
      message,
      actionUrl,
      sendEmail: true,
      sendPush: true,
    });

    this.logger.logSecurity('security_alert_sent', userId, undefined, {
      title,
      message,
    });
  }

  async sendVideoProcessedNotification(
    userId: string,
    videoId: string,
    videoTitle: string,
    success: boolean,
  ): Promise<void> {
    const title = success ? 'Video Ready!' : 'Video Processing Failed';
    const message = success 
      ? `Your video "${videoTitle}" is now ready to watch!`
      : `There was an error processing your video "${videoTitle}". Please try uploading again.`;

    await this.sendNotification({
      userId,
      type: success ? NotificationType.VIDEO_PROCESSED : NotificationType.VIDEO_UPLOADED,
      title,
      message,
      actionUrl: success ? `/videos/${videoId}` : '/upload',
      relatedId: videoId,
      sendEmail: true,
      sendPush: true,
    });
  }

  async sendPaymentNotification(
    userId: string,
    amount: number,
    currency: string,
    success: boolean,
    paymentId: string,
  ): Promise<void> {
    const title = success ? 'Payment Received!' : 'Payment Failed';
    const message = success
      ? `Your payment of ${currency.toUpperCase()} ${(amount / 100).toFixed(2)} has been processed successfully.`
      : `Your payment of ${currency.toUpperCase()} ${(amount / 100).toFixed(2)} could not be processed. Please try again.`;

    await this.sendNotification({
      userId,
      type: success ? NotificationType.PAYMENT_RECEIVED : NotificationType.PAYMENT_FAILED,
      title,
      message,
      actionUrl: '/payments',
      relatedId: paymentId,
      sendEmail: true,
      sendPush: true,
    });
  }

  async sendSubscriptionNotification(
    userId: string,
    plan: string,
    success: boolean,
    subscriptionId: string,
  ): Promise<void> {
    const title = success ? 'Subscription Created!' : 'Subscription Failed';
    const message = success
      ? `Your ${plan} subscription has been activated successfully.`
      : `There was an error creating your ${plan} subscription. Please try again.`;

    await this.sendNotification({
      userId,
      type: success ? NotificationType.SUBSCRIPTION_CREATED : NotificationType.SUBSCRIPTION_CANCELLED,
      title,
      message,
      actionUrl: '/subscriptions',
      relatedId: subscriptionId,
      sendEmail: true,
      sendPush: true,
    });
  }

  async sendAdNotification(
    userId: string,
    adId: string,
    adTitle: string,
    approved: boolean,
  ): Promise<void> {
    const title = approved ? 'Ad Approved!' : 'Ad Rejected';
    const message = approved
      ? `Your ad "${adTitle}" has been approved and is now live.`
      : `Your ad "${adTitle}" was rejected. Please review the guidelines and try again.`;

    await this.sendNotification({
      userId,
      type: approved ? NotificationType.AD_APPROVED : NotificationType.AD_REJECTED,
      title,
      message,
      actionUrl: '/ads',
      relatedId: adId,
      sendEmail: true,
      sendPush: true,
    });
  }

  async cleanupOldNotifications(days: number = 30): Promise<void> {
    await this.notificationRepository.deleteOldNotifications(days);

    this.logger.logBusiness('old_notifications_cleaned', undefined, {
      days,
    });
  }
}
