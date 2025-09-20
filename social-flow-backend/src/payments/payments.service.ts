import { Injectable, NotFoundException, ForbiddenException, BadRequestException } from '@nestjs/common';
import { PaymentRepository } from '../shared/database/repositories/payment.repository';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { Payment, PaymentStatus, PaymentMethod, PaymentType } from '../shared/database/entities/payment.entity';
import { LoggerService } from '../shared/logger/logger.service';
import Stripe from 'stripe';

export interface CreatePaymentRequest {
  amount: number;
  currency: string;
  method: PaymentMethod;
  type: PaymentType;
  description?: string;
  metadata?: Record<string, any>;
}

export interface ProcessPaymentRequest {
  paymentId: string;
  paymentMethodId: string;
  customerId?: string;
}

@Injectable()
export class PaymentsService {
  private stripe: Stripe;

  constructor(
    private paymentRepository: PaymentRepository,
    private userRepository: UserRepository,
    private logger: LoggerService,
  ) {
    this.stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
      apiVersion: '2023-10-16',
    });
  }

  async createPayment(userId: string, paymentData: CreatePaymentRequest): Promise<Payment> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    if (paymentData.amount <= 0) {
      throw new BadRequestException('Amount must be greater than 0');
    }

    const payment = await this.paymentRepository.create({
      amount: paymentData.amount,
      currency: paymentData.currency,
      method: paymentData.method,
      type: paymentData.type,
      description: paymentData.description,
      metadata: paymentData.metadata,
      userId,
      status: PaymentStatus.PENDING,
    });

    this.logger.logBusiness('payment_created', userId, {
      paymentId: payment.id,
      amount: paymentData.amount,
      currency: paymentData.currency,
      type: paymentData.type,
    });

    return payment;
  }

  async processPayment(paymentId: string, userId: string, processData: ProcessPaymentRequest): Promise<Payment> {
    const payment = await this.paymentRepository.findById(paymentId);
    if (!payment) {
      throw new NotFoundException('Payment not found');
    }

    if (payment.userId !== userId) {
      throw new ForbiddenException('Not authorized to process this payment');
    }

    if (payment.status !== PaymentStatus.PENDING) {
      throw new BadRequestException('Payment is not in pending status');
    }

    try {
      // Update payment status to processing
      await this.paymentRepository.updateStatus(paymentId, PaymentStatus.PROCESSING);

      // Process payment based on method
      let result: any;
      if (payment.method === PaymentMethod.STRIPE) {
        result = await this.processStripePayment(payment, processData);
      } else {
        throw new BadRequestException('Unsupported payment method');
      }

      // Update payment with external ID and mark as completed
      await this.paymentRepository.update(paymentId, {
        externalId: result.id,
        status: PaymentStatus.COMPLETED,
        paymentData: result,
      });

      this.logger.logBusiness('payment_completed', userId, {
        paymentId,
        externalId: result.id,
        amount: payment.amount,
      });

      return payment;
    } catch (error) {
      // Update payment status to failed
      await this.paymentRepository.updateStatus(paymentId, PaymentStatus.FAILED, error.message);

      this.logger.logError(error, 'PaymentsService.processPayment', {
        paymentId,
        userId,
      });

      throw error;
    }
  }

  async getPayment(paymentId: string, userId: string): Promise<Payment> {
    const payment = await this.paymentRepository.findById(paymentId);
    if (!payment) {
      throw new NotFoundException('Payment not found');
    }

    if (payment.userId !== userId) {
      throw new ForbiddenException('Not authorized to view this payment');
    }

    return payment;
  }

  async getUserPayments(userId: string, limit: number = 10, offset: number = 0): Promise<Payment[]> {
    return this.paymentRepository.findByUser(userId, limit, offset);
  }

  async getPaymentsByStatus(status: PaymentStatus, limit: number = 10, offset: number = 0): Promise<Payment[]> {
    return this.paymentRepository.findByStatus(status, limit, offset);
  }

  async getPaymentsByType(type: PaymentType, limit: number = 10, offset: number = 0): Promise<Payment[]> {
    return this.paymentRepository.findByType(type, limit, offset);
  }

  async refundPayment(paymentId: string, userId: string, reason?: string): Promise<Payment> {
    const payment = await this.paymentRepository.findById(paymentId);
    if (!payment) {
      throw new NotFoundException('Payment not found');
    }

    if (payment.userId !== userId) {
      throw new ForbiddenException('Not authorized to refund this payment');
    }

    if (payment.status !== PaymentStatus.COMPLETED) {
      throw new BadRequestException('Payment must be completed to refund');
    }

    try {
      let refundResult: any;
      if (payment.method === PaymentMethod.STRIPE) {
        refundResult = await this.stripe.refunds.create({
          payment_intent: payment.externalId,
          reason: 'requested_by_customer',
          metadata: {
            reason: reason || 'No reason provided',
          },
        });
      } else {
        throw new BadRequestException('Unsupported payment method for refund');
      }

      // Update payment with refund information
      await this.paymentRepository.update(paymentId, {
        status: PaymentStatus.REFUNDED,
        refundId: refundResult.id,
        refundAmount: refundResult.amount,
        refundReason: reason,
      });

      this.logger.logBusiness('payment_refunded', userId, {
        paymentId,
        refundId: refundResult.id,
        refundAmount: refundResult.amount,
        reason,
      });

      return payment;
    } catch (error) {
      this.logger.logError(error, 'PaymentsService.refundPayment', {
        paymentId,
        userId,
      });
      throw error;
    }
  }

  async getTotalRevenue(userId?: string, startDate?: Date, endDate?: Date): Promise<number> {
    return this.paymentRepository.getTotalRevenue(userId, startDate, endDate);
  }

  async getPaymentStats(userId: string): Promise<{
    totalPayments: number;
    totalAmount: number;
    successfulPayments: number;
    failedPayments: number;
    refundedPayments: number;
    averagePaymentAmount: number;
  }> {
    const payments = await this.paymentRepository.findByUser(userId, 1000, 0);
    
    const totalPayments = payments.length;
    const totalAmount = payments.reduce((sum, payment) => sum + payment.amount, 0);
    const successfulPayments = payments.filter(p => p.status === PaymentStatus.COMPLETED).length;
    const failedPayments = payments.filter(p => p.status === PaymentStatus.FAILED).length;
    const refundedPayments = payments.filter(p => p.status === PaymentStatus.REFUNDED).length;
    const averagePaymentAmount = totalPayments > 0 ? totalAmount / totalPayments : 0;

    return {
      totalPayments,
      totalAmount,
      successfulPayments,
      failedPayments,
      refundedPayments,
      averagePaymentAmount,
    };
  }

  private async processStripePayment(payment: Payment, processData: ProcessPaymentRequest): Promise<any> {
    const paymentIntent = await this.stripe.paymentIntents.create({
      amount: payment.amount,
      currency: payment.currency,
      payment_method: processData.paymentMethodId,
      customer: processData.customerId,
      confirmation_method: 'manual',
      confirm: true,
      description: payment.description,
      metadata: {
        paymentId: payment.id,
        userId: payment.userId,
        ...payment.metadata,
      },
    });

    return paymentIntent;
  }

  async createStripeCustomer(userId: string, email: string, name?: string): Promise<string> {
    const customer = await this.stripe.customers.create({
      email,
      name,
      metadata: {
        userId,
      },
    });

    this.logger.logBusiness('stripe_customer_created', userId, {
      customerId: customer.id,
      email,
    });

    return customer.id;
  }

  async createStripePaymentMethod(userId: string, cardToken: string): Promise<string> {
    const paymentMethod = await this.stripe.paymentMethods.create({
      type: 'card',
      card: {
        token: cardToken,
      },
    });

    this.logger.logBusiness('stripe_payment_method_created', userId, {
      paymentMethodId: paymentMethod.id,
    });

    return paymentMethod.id;
  }

  async getStripePaymentMethods(userId: string, customerId: string): Promise<any[]> {
    const paymentMethods = await this.stripe.paymentMethods.list({
      customer: customerId,
      type: 'card',
    });

    return paymentMethods.data;
  }

  async deleteStripePaymentMethod(paymentMethodId: string): Promise<void> {
    await this.stripe.paymentMethods.detach(paymentMethodId);
  }
}
