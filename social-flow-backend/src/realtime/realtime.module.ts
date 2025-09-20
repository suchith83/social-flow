import { Module } from '@nestjs/common';
import { DatabaseModule } from '../shared/database/database.module';
import { RedisModule } from '../shared/redis/redis.module';
import { RealtimeService } from './realtime.service';
import { RealtimeGateway } from './realtime.gateway';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { NotificationRepository } from '../shared/database/repositories/notification.repository';

@Module({
  imports: [DatabaseModule, RedisModule],
  providers: [
    RealtimeService,
    RealtimeGateway,
    UserRepository,
    NotificationRepository,
  ],
})
export class RealtimeModule {}
