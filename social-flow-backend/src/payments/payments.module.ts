import { Module } from '@nestjs/common';
import { PaymentsController } from './payments.controller';
import { PaymentsService } from './payments.service';
import { SubscriptionsService } from './subscriptions.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Module({
  controllers: [PaymentsController],
  providers: [PaymentsService, SubscriptionsService, JwtAuthGuard],
  exports: [PaymentsService, SubscriptionsService],
})
export class PaymentsModule {}
