import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bull';
import { RedisService } from './redis.service';

@Module({
  imports: [
    BullModule.registerQueue(
      { name: 'video-processing' },
      { name: 'email' },
      { name: 'push-notification' },
      { name: 'analytics' },
      { name: 'cleanup' },
    ),
  ],
  providers: [RedisService],
  exports: [RedisService, BullModule],
})
export class RedisModule {}
