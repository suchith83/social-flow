import {
  Controller,
  Get,
  Post,
  Put,
  Delete,
  Body,
  Param,
  Query,
  UseGuards,
  Request,
  ValidationPipe,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { ThrottlerGuard } from '@nestjs/throttler';

import { PaymentsService, CreatePaymentRequest, ProcessPaymentRequest } from './payments.service';
import { SubscriptionsService, CreateSubscriptionRequest, UpdateSubscriptionRequest } from './subscriptions.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('payments')
@Controller('payments')
@UseGuards(ThrottlerGuard)
export class PaymentsController {
  constructor(
    private readonly paymentsService: PaymentsService,
    private readonly subscriptionsService: SubscriptionsService,
  ) {}

  // Payment endpoints
  @Post()
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Create a new payment' })
  @ApiResponse({ status: 201, description: 'Payment created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  async createPayment(
    @Body(ValidationPipe) paymentData: CreatePaymentRequest,
    @Request() req,
  ) {
    return this.paymentsService.createPayment(req.user.id, paymentData);
  }

  @Post(':id/process')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Process payment' })
  @ApiResponse({ status: 200, description: 'Payment processed successfully' })
  @ApiResponse({ status: 404, description: 'Payment not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to process this payment' })
  async processPayment(
    @Param('id') paymentId: string,
    @Body(ValidationPipe) processData: ProcessPaymentRequest,
    @Request() req,
  ) {
    return this.paymentsService.processPayment(paymentId, req.user.id, processData);
  }

  @Get(':id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get payment by ID' })
  @ApiResponse({ status: 200, description: 'Payment retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Payment not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to view this payment' })
  async getPayment(@Param('id') paymentId: string, @Request() req) {
    return this.paymentsService.getPayment(paymentId, req.user.id);
  }

  @Get()
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get user payments' })
  @ApiResponse({ status: 200, description: 'Payments retrieved successfully' })
  async getUserPayments(
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
    @Request() req,
  ) {
    return this.paymentsService.getUserPayments(req.user.id, limit, offset);
  }

  @Post(':id/refund')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Refund payment' })
  @ApiResponse({ status: 200, description: 'Payment refunded successfully' })
  @ApiResponse({ status: 404, description: 'Payment not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to refund this payment' })
  async refundPayment(
    @Param('id') paymentId: string,
    @Body('reason') reason: string,
    @Request() req,
  ) {
    return this.paymentsService.refundPayment(paymentId, req.user.id, reason);
  }

  @Get('stats/revenue')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get revenue statistics' })
  @ApiResponse({ status: 200, description: 'Revenue stats retrieved successfully' })
  async getRevenueStats(
    @Query('startDate') startDate?: string,
    @Query('endDate') endDate?: string,
    @Request() req,
  ) {
    const start = startDate ? new Date(startDate) : undefined;
    const end = endDate ? new Date(endDate) : undefined;
    return this.paymentsService.getTotalRevenue(req.user.id, start, end);
  }

  @Get('stats/payments')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get payment statistics' })
  @ApiResponse({ status: 200, description: 'Payment stats retrieved successfully' })
  async getPaymentStats(@Request() req) {
    return this.paymentsService.getPaymentStats(req.user.id);
  }

  // Stripe integration endpoints
  @Post('stripe/customer')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Create Stripe customer' })
  @ApiResponse({ status: 201, description: 'Customer created successfully' })
  async createStripeCustomer(
    @Body('email') email: string,
    @Body('name') name: string,
    @Request() req,
  ) {
    const customerId = await this.paymentsService.createStripeCustomer(req.user.id, email, name);
    return { customerId };
  }

  @Post('stripe/payment-method')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Create Stripe payment method' })
  @ApiResponse({ status: 201, description: 'Payment method created successfully' })
  async createStripePaymentMethod(
    @Body('cardToken') cardToken: string,
    @Request() req,
  ) {
    const paymentMethodId = await this.paymentsService.createStripePaymentMethod(req.user.id, cardToken);
    return { paymentMethodId };
  }

  @Get('stripe/payment-methods')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get Stripe payment methods' })
  @ApiResponse({ status: 200, description: 'Payment methods retrieved successfully' })
  async getStripePaymentMethods(
    @Query('customerId') customerId: string,
    @Request() req,
  ) {
    return this.paymentsService.getStripePaymentMethods(req.user.id, customerId);
  }

  @Delete('stripe/payment-method/:paymentMethodId')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Delete Stripe payment method' })
  @ApiResponse({ status: 200, description: 'Payment method deleted successfully' })
  async deleteStripePaymentMethod(@Param('paymentMethodId') paymentMethodId: string) {
    return this.paymentsService.deleteStripePaymentMethod(paymentMethodId);
  }

  // Subscription endpoints
  @Post('subscriptions')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Create a new subscription' })
  @ApiResponse({ status: 201, description: 'Subscription created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  async createSubscription(
    @Body(ValidationPipe) subscriptionData: CreateSubscriptionRequest,
    @Request() req,
  ) {
    return this.subscriptionsService.createSubscription(req.user.id, subscriptionData);
  }

  @Get('subscriptions/:id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get subscription by ID' })
  @ApiResponse({ status: 200, description: 'Subscription retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Subscription not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to view this subscription' })
  async getSubscription(@Param('id') subscriptionId: string, @Request() req) {
    return this.subscriptionsService.getSubscription(subscriptionId, req.user.id);
  }

  @Get('subscriptions')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get user subscription' })
  @ApiResponse({ status: 200, description: 'Subscription retrieved successfully' })
  async getUserSubscription(@Request() req) {
    return this.subscriptionsService.getUserSubscription(req.user.id);
  }

  @Put('subscriptions/:id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update subscription' })
  @ApiResponse({ status: 200, description: 'Subscription updated successfully' })
  @ApiResponse({ status: 404, description: 'Subscription not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to update this subscription' })
  async updateSubscription(
    @Param('id') subscriptionId: string,
    @Body(ValidationPipe) updateData: UpdateSubscriptionRequest,
    @Request() req,
  ) {
    return this.subscriptionsService.updateSubscription(subscriptionId, req.user.id, updateData);
  }

  @Post('subscriptions/:id/cancel')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Cancel subscription' })
  @ApiResponse({ status: 200, description: 'Subscription cancelled successfully' })
  @ApiResponse({ status: 404, description: 'Subscription not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to cancel this subscription' })
  async cancelSubscription(
    @Param('id') subscriptionId: string,
    @Body('immediately') immediately: boolean = false,
    @Request() req,
  ) {
    return this.subscriptionsService.cancelSubscription(subscriptionId, req.user.id, immediately);
  }

  @Post('subscriptions/:id/reactivate')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Reactivate subscription' })
  @ApiResponse({ status: 200, description: 'Subscription reactivated successfully' })
  @ApiResponse({ status: 404, description: 'Subscription not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to reactivate this subscription' })
  async reactivateSubscription(@Param('id') subscriptionId: string, @Request() req) {
    return this.subscriptionsService.reactivateSubscription(subscriptionId, req.user.id);
  }

  @Get('subscriptions/stats/overview')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get subscription statistics' })
  @ApiResponse({ status: 200, description: 'Subscription stats retrieved successfully' })
  async getSubscriptionStats(@Request() req) {
    return this.subscriptionsService.getSubscriptionStats();
  }
}
