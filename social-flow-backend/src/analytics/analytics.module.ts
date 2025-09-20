import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bull';
import { AnalyticsController } from './analytics.controller';
import { AnalyticsService } from './analytics.service';
import { AnalyticsProcessor } from './processors/analytics.processor';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Module({
  imports: [
    BullModule.registerQueue({ name: 'analytics' }),
  ],
  controllers: [AnalyticsController],
  providers: [AnalyticsService, AnalyticsProcessor, JwtAuthGuard],
  exports: [AnalyticsService],
})
export class AnalyticsModule {}
