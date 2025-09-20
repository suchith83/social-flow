import { Process, Processor } from '@nestjs/bull';
import { Job } from 'bull';
import { LoggerService } from '../../shared/logger/logger.service';
import { PushNotificationService } from '../push-notification.service';

@Processor('push-notification')
export class PushNotificationProcessor {
  constructor(
    private logger: LoggerService,
    private pushNotificationService: PushNotificationService,
  ) {}

  @Process('send-push')
  async handlePushNotification(job: Job<any>) {
    const { userId, title, body, data, imageUrl } = job.data;

    try {
      this.logger.logBusiness('push_notification_processing_started', userId, {
        title,
        body,
        jobId: job.id,
      });

      await this.pushNotificationService.sendPushNotification(userId, {
        title,
        body,
        data,
        imageUrl,
      });

      this.logger.logBusiness('push_notification_processing_completed', userId, {
        title,
        body,
        jobId: job.id,
      });
    } catch (error) {
      this.logger.logError(error, 'PushNotificationProcessor.handlePushNotification', {
        userId,
        title,
        body,
        jobId: job.id,
      });
      throw error;
    }
  }

  @Process('send-bulk-push')
  async handleBulkPushNotification(job: Job<any>) {
    const { userIds, title, body, data, imageUrl } = job.data;

    try {
      this.logger.logBusiness('bulk_push_notification_processing_started', undefined, {
        userIds: userIds.length,
        title,
        body,
        jobId: job.id,
      });

      await this.pushNotificationService.sendBulkPushNotification(userIds, {
        title,
        body,
        data,
        imageUrl,
      });

      this.logger.logBusiness('bulk_push_notification_processing_completed', undefined, {
        userIds: userIds.length,
        title,
        body,
        jobId: job.id,
      });
    } catch (error) {
      this.logger.logError(error, 'PushNotificationProcessor.handleBulkPushNotification', {
        userIds,
        title,
        body,
        jobId: job.id,
      });
      throw error;
    }
  }

  @Process('send-topic-push')
  async handleTopicPushNotification(job: Job<any>) {
    const { topic, title, body, data, imageUrl } = job.data;

    try {
      this.logger.logBusiness('topic_push_notification_processing_started', undefined, {
        topic,
        title,
        body,
        jobId: job.id,
      });

      await this.pushNotificationService.sendTopicNotification(topic, {
        title,
        body,
        data,
        imageUrl,
      });

      this.logger.logBusiness('topic_push_notification_processing_completed', undefined, {
        topic,
        title,
        body,
        jobId: job.id,
      });
    } catch (error) {
      this.logger.logError(error, 'PushNotificationProcessor.handleTopicPushNotification', {
        topic,
        title,
        body,
        jobId: job.id,
      });
      throw error;
    }
  }
}
