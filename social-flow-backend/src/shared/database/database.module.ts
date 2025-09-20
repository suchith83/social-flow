import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';

// Entities
import { User } from './entities/user.entity';
import { Video } from './entities/video.entity';
import { Post } from './entities/post.entity';
import { Comment } from './entities/comment.entity';
import { Like } from './entities/like.entity';
import { Follow } from './entities/follow.entity';
import { Ad } from './entities/ad.entity';
import { Payment } from './entities/payment.entity';
import { Subscription } from './entities/subscription.entity';
import { Notification } from './entities/notification.entity';
import { Analytics } from './entities/analytics.entity';
import { ViewCount } from './entities/view-count.entity';

// Repositories
import { UserRepository } from './repositories/user.repository';
import { VideoRepository } from './repositories/video.repository';
import { PostRepository } from './repositories/post.repository';
import { CommentRepository } from './repositories/comment.repository';
import { LikeRepository } from './repositories/like.repository';
import { FollowRepository } from './repositories/follow.repository';
import { AdRepository } from './repositories/ad.repository';
import { PaymentRepository } from './repositories/payment.repository';
import { SubscriptionRepository } from './repositories/subscription.repository';
import { NotificationRepository } from './repositories/notification.repository';
import { AnalyticsRepository } from './repositories/analytics.repository';
import { ViewCountRepository } from './repositories/view-count.repository';

@Module({
  imports: [
    TypeOrmModule.forFeature([
      User,
      Video,
      Post,
      Comment,
      Like,
      Follow,
      Ad,
      Payment,
      Subscription,
      Notification,
      Analytics,
      ViewCount,
    ]),
  ],
  providers: [
    UserRepository,
    VideoRepository,
    PostRepository,
    CommentRepository,
    LikeRepository,
    FollowRepository,
    AdRepository,
    PaymentRepository,
    SubscriptionRepository,
    NotificationRepository,
    AnalyticsRepository,
    ViewCountRepository,
  ],
  exports: [
    UserRepository,
    VideoRepository,
    PostRepository,
    CommentRepository,
    LikeRepository,
    FollowRepository,
    AdRepository,
    PaymentRepository,
    SubscriptionRepository,
    NotificationRepository,
    AnalyticsRepository,
    ViewCountRepository,
  ],
})
export class DatabaseModule {}
