import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bull';
import { NotificationsController } from './notifications.controller';
import { NotificationsService } from './notifications.service';
import { EmailService } from './email.service';
import { PushNotificationService } from './push-notification.service';
import { EmailProcessor } from './processors/email.processor';
import { PushNotificationProcessor } from './processors/push-notification.processor';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Module({
  imports: [
    BullModule.registerQueue(
      { name: 'email' },
      { name: 'push-notification' },
    ),
  ],
  controllers: [NotificationsController],
  providers: [
    NotificationsService,
    EmailService,
    PushNotificationService,
    EmailProcessor,
    PushNotificationProcessor,
    JwtAuthGuard,
  ],
  exports: [NotificationsService, EmailService, PushNotificationService],
})
export class NotificationsModule {}
