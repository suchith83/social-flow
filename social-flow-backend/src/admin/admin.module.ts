import { Module } from '@nestjs/common';
import { DatabaseModule } from '../shared/database/database.module';
import { AwsModule } from '../shared/aws/aws.module';
import { AdminService } from './admin.service';
import { AdminController } from './admin.controller';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { VideoRepository } from '../shared/database/repositories/video.repository';
import { PostRepository } from '../shared/database/repositories/post.repository';
import { AdRepository } from '../shared/database/repositories/ad.repository';
import { PaymentRepository } from '../shared/database/repositories/payment.repository';
import { SubscriptionRepository } from '../shared/database/repositories/subscription.repository';
import { NotificationRepository } from '../shared/database/repositories/notification.repository';
import { AnalyticsRepository } from '../shared/database/repositories/analytics.repository';
import { ViewCountRepository } from '../shared/database/repositories/view-count.repository';

@Module({
  imports: [DatabaseModule, AwsModule],
  providers: [
    AdminService,
    UserRepository,
    VideoRepository,
    PostRepository,
    AdRepository,
    PaymentRepository,
    SubscriptionRepository,
    NotificationRepository,
    AnalyticsRepository,
    ViewCountRepository,
  ],
  controllers: [AdminController],
})
export class AdminModule {}
