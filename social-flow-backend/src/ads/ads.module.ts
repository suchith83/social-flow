import { Module } from '@nestjs/common';
import { AdsController } from './ads.controller';
import { AdsService } from './ads.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Module({
  controllers: [AdsController],
  providers: [AdsService, JwtAuthGuard],
  exports: [AdsService],
})
export class AdsModule {}
