import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Notification, NotificationStatus, NotificationType } from '../entities/notification.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class NotificationRepository extends BaseRepository<Notification> {
  constructor(
    @InjectRepository(Notification)
    private readonly notificationRepository: Repository<Notification>,
  ) {
    super(notificationRepository);
  }

  async findByUser(userId: string, limit: number = 10, offset: number = 0): Promise<Notification[]> {
    return this.notificationRepository.find({
      where: { userId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findUnreadByUser(userId: string, limit: number = 10, offset: number = 0): Promise<Notification[]> {
    return this.notificationRepository.find({
      where: { userId, status: NotificationStatus.UNREAD },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByType(type: NotificationType, limit: number = 10, offset: number = 0): Promise<Notification[]> {
    return this.notificationRepository.find({
      where: { type },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async markAsRead(notificationId: string): Promise<void> {
    await this.notificationRepository.update(notificationId, {
      status: NotificationStatus.READ,
      isRead: true,
      readAt: new Date(),
    });
  }

  async markAllAsRead(userId: string): Promise<void> {
    await this.notificationRepository.update(
      { userId, status: NotificationStatus.UNREAD },
      {
        status: NotificationStatus.READ,
        isRead: true,
        readAt: new Date(),
      }
    );
  }

  async countUnreadByUser(userId: string): Promise<number> {
    return this.notificationRepository.count({
      where: { userId, status: NotificationStatus.UNREAD },
    });
  }

  async deleteOldNotifications(days: number = 30): Promise<void> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - days);

    await this.notificationRepository
      .createQueryBuilder()
      .delete()
      .where('createdAt < :cutoffDate', { cutoffDate })
      .execute();
  }
}
