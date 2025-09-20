import { Injectable, Inject } from '@nestjs/common';
import { InjectQueue } from '@nestjs/bull';
import { Queue } from 'bull';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class RedisService {
  constructor(
    @InjectQueue('video-processing') private videoProcessingQueue: Queue,
    @InjectQueue('email') private emailQueue: Queue,
    @InjectQueue('push-notification') private pushNotificationQueue: Queue,
    @InjectQueue('analytics') private analyticsQueue: Queue,
    @InjectQueue('cleanup') private cleanupQueue: Queue,
    private configService: ConfigService,
  ) {}

  // Video Processing Queue
  async addVideoProcessingJob(data: {
    videoId: string;
    filePath: string;
    userId: string;
    metadata: Record<string, any>;
  }) {
    return this.videoProcessingQueue.add('process-video', data, {
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 2000,
      },
      removeOnComplete: 10,
      removeOnFail: 5,
    });
  }

  async addThumbnailGenerationJob(data: {
    videoId: string;
    filePath: string;
    timestamps: number[];
  }) {
    return this.videoProcessingQueue.add('generate-thumbnails', data, {
      attempts: 2,
      backoff: {
        type: 'exponential',
        delay: 1000,
      },
      removeOnComplete: 10,
      removeOnFail: 5,
    });
  }

  // Email Queue
  async addEmailJob(data: {
    to: string;
    subject: string;
    template: string;
    context: Record<string, any>;
  }) {
    return this.emailQueue.add('send-email', data, {
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 2000,
      },
      removeOnComplete: 10,
      removeOnFail: 5,
    });
  }

  // Push Notification Queue
  async addPushNotificationJob(data: {
    userId: string;
    title: string;
    body: string;
    data?: Record<string, any>;
    imageUrl?: string;
  }) {
    return this.pushNotificationQueue.add('send-push', data, {
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 2000,
      },
      removeOnComplete: 10,
      removeOnFail: 5,
    });
  }

  // Analytics Queue
  async addAnalyticsJob(data: {
    type: string;
    category: string;
    userId?: string;
    entityType?: string;
    entityId?: string;
    event: string;
    properties?: Record<string, any>;
    context?: Record<string, any>;
  }) {
    return this.analyticsQueue.add('track-event', data, {
      attempts: 2,
      backoff: {
        type: 'exponential',
        delay: 1000,
      },
      removeOnComplete: 100,
      removeOnFail: 10,
    });
  }

  // Cleanup Queue
  async addCleanupJob(data: {
    type: string;
    entityId: string;
    retentionDays: number;
  }) {
    return this.cleanupQueue.add('cleanup-data', data, {
      attempts: 1,
      removeOnComplete: 5,
      removeOnFail: 5,
    });
  }

  // Queue Management
  async getQueueStats() {
    return {
      videoProcessing: {
        waiting: await this.videoProcessingQueue.getWaiting(),
        active: await this.videoProcessingQueue.getActive(),
        completed: await this.videoProcessingQueue.getCompleted(),
        failed: await this.videoProcessingQueue.getFailed(),
      },
      email: {
        waiting: await this.emailQueue.getWaiting(),
        active: await this.emailQueue.getActive(),
        completed: await this.emailQueue.getCompleted(),
        failed: await this.emailQueue.getFailed(),
      },
      pushNotification: {
        waiting: await this.pushNotificationQueue.getWaiting(),
        active: await this.pushNotificationQueue.getActive(),
        completed: await this.pushNotificationQueue.getCompleted(),
        failed: await this.pushNotificationQueue.getFailed(),
      },
      analytics: {
        waiting: await this.analyticsQueue.getWaiting(),
        active: await this.analyticsQueue.getActive(),
        completed: await this.analyticsQueue.getCompleted(),
        failed: await this.analyticsQueue.getFailed(),
      },
      cleanup: {
        waiting: await this.cleanupQueue.getWaiting(),
        active: await this.cleanupQueue.getActive(),
        completed: await this.cleanupQueue.getCompleted(),
        failed: await this.cleanupQueue.getFailed(),
      },
    };
  }

  async clearQueue(queueName: string) {
    const queue = this.getQueueByName(queueName);
    if (queue) {
      await queue.empty();
    }
  }

  async pauseQueue(queueName: string) {
    const queue = this.getQueueByName(queueName);
    if (queue) {
      await queue.pause();
    }
  }

  async resumeQueue(queueName: string) {
    const queue = this.getQueueByName(queueName);
    if (queue) {
      await queue.resume();
    }
  }

  private getQueueByName(queueName: string): Queue | null {
    switch (queueName) {
      case 'video-processing':
        return this.videoProcessingQueue;
      case 'email':
        return this.emailQueue;
      case 'push-notification':
        return this.pushNotificationQueue;
      case 'analytics':
        return this.analyticsQueue;
      case 'cleanup':
        return this.cleanupQueue;
      default:
        return null;
    }
  }
}
