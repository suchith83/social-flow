import { Injectable, NotFoundException, ForbiddenException, BadRequestException } from '@nestjs/common';
import { SubscriptionRepository } from '../shared/database/repositories/subscription.repository';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { Subscription, SubscriptionStatus, SubscriptionPlan } from '../shared/database/entities/subscription.entity';
import { LoggerService } from '../shared/logger/logger.service';
import Stripe from 'stripe';

export interface CreateSubscriptionRequest {
  plan: SubscriptionPlan;
  paymentMethodId: string;
  customerId?: string;
}

export interface UpdateSubscriptionRequest {
  plan?: SubscriptionPlan;
  paymentMethodId?: string;
}

@Injectable()
export class SubscriptionsService {
  private stripe: Stripe;

  constructor(
    private subscriptionRepository: SubscriptionRepository,
    private userRepository: UserRepository,
    private logger: LoggerService,
  ) {
    this.stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
      apiVersion: '2023-10-16',
    });
  }

  async createSubscription(userId: string, subscriptionData: CreateSubscriptionRequest): Promise<Subscription> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Check if user already has an active subscription
    const existingSubscription = await this.subscriptionRepository.findByUser(userId);
    if (existingSubscription && existingSubscription.isActive) {
      throw new BadRequestException('User already has an active subscription');
    }

    // Get plan details
    const planDetails = this.getPlanDetails(subscriptionData.plan);
    if (!planDetails) {
      throw new BadRequestException('Invalid subscription plan');
    }

    try {
      // Create Stripe subscription
      const stripeSubscription = await this.stripe.subscriptions.create({
        customer: subscriptionData.customerId,
        items: [{
          price: planDetails.priceId,
        }],
        payment_behavior: 'default_incomplete',
        payment_settings: {
          save_default_payment_method: 'on_subscription',
        },
        expand: ['latest_invoice.payment_intent'],
        metadata: {
          userId,
        },
      });

      // Create subscription record
      const subscription = await this.subscriptionRepository.create({
        plan: subscriptionData.plan,
        status: SubscriptionStatus.INCOMPLETE,
        externalId: stripeSubscription.id,
        priceId: planDetails.priceId,
        amount: planDetails.amount,
        currency: planDetails.currency,
        interval: planDetails.interval,
        intervalCount: planDetails.intervalCount,
        features: planDetails.features,
        userId,
      });

      this.logger.logBusiness('subscription_created', userId, {
        subscriptionId: subscription.id,
        plan: subscriptionData.plan,
        stripeSubscriptionId: stripeSubscription.id,
      });

      return subscription;
    } catch (error) {
      this.logger.logError(error, 'SubscriptionsService.createSubscription', {
        userId,
        plan: subscriptionData.plan,
      });
      throw error;
    }
  }

  async getSubscription(subscriptionId: string, userId: string): Promise<Subscription> {
    const subscription = await this.subscriptionRepository.findById(subscriptionId);
    if (!subscription) {
      throw new NotFoundException('Subscription not found');
    }

    if (subscription.userId !== userId) {
      throw new ForbiddenException('Not authorized to view this subscription');
    }

    return subscription;
  }

  async getUserSubscription(userId: string): Promise<Subscription | null> {
    return this.subscriptionRepository.findByUser(userId);
  }

  async updateSubscription(subscriptionId: string, userId: string, updateData: UpdateSubscriptionRequest): Promise<Subscription> {
    const subscription = await this.subscriptionRepository.findById(subscriptionId);
    if (!subscription) {
      throw new NotFoundException('Subscription not found');
    }

    if (subscription.userId !== userId) {
      throw new ForbiddenException('Not authorized to update this subscription');
    }

    try {
      if (updateData.plan) {
        const planDetails = this.getPlanDetails(updateData.plan);
        if (!planDetails) {
          throw new BadRequestException('Invalid subscription plan');
        }

        // Update Stripe subscription
        await this.stripe.subscriptions.update(subscription.externalId, {
          items: [{
            id: subscription.priceId,
            price: planDetails.priceId,
          }],
          proration_behavior: 'create_prorations',
        });

        // Update subscription record
        await this.subscriptionRepository.update(subscriptionId, {
          plan: updateData.plan,
          priceId: planDetails.priceId,
          amount: planDetails.amount,
          features: planDetails.features,
        });
      }

      if (updateData.paymentMethodId) {
        // Update default payment method
        await this.stripe.subscriptions.update(subscription.externalId, {
          default_payment_method: updateData.paymentMethodId,
        });
      }

      this.logger.logBusiness('subscription_updated', userId, {
        subscriptionId,
        updates: updateData,
      });

      return subscription;
    } catch (error) {
      this.logger.logError(error, 'SubscriptionsService.updateSubscription', {
        subscriptionId,
        userId,
        updates: updateData,
      });
      throw error;
    }
  }

  async cancelSubscription(subscriptionId: string, userId: string, immediately: boolean = false): Promise<Subscription> {
    const subscription = await this.subscriptionRepository.findById(subscriptionId);
    if (!subscription) {
      throw new NotFoundException('Subscription not found');
    }

    if (subscription.userId !== userId) {
      throw new ForbiddenException('Not authorized to cancel this subscription');
    }

    try {
      if (immediately) {
        // Cancel immediately
        await this.stripe.subscriptions.cancel(subscription.externalId);
        await this.subscriptionRepository.updateStatus(subscriptionId, SubscriptionStatus.CANCELLED);
      } else {
        // Cancel at period end
        await this.stripe.subscriptions.update(subscription.externalId, {
          cancel_at_period_end: true,
        });
        await this.subscriptionRepository.update(subscriptionId, {
          cancelAtPeriodEnd: true,
        });
      }

      this.logger.logBusiness('subscription_cancelled', userId, {
        subscriptionId,
        immediately,
      });

      return subscription;
    } catch (error) {
      this.logger.logError(error, 'SubscriptionsService.cancelSubscription', {
        subscriptionId,
        userId,
        immediately,
      });
      throw error;
    }
  }

  async reactivateSubscription(subscriptionId: string, userId: string): Promise<Subscription> {
    const subscription = await this.subscriptionRepository.findById(subscriptionId);
    if (!subscription) {
      throw new NotFoundException('Subscription not found');
    }

    if (subscription.userId !== userId) {
      throw new ForbiddenException('Not authorized to reactivate this subscription');
    }

    try {
      // Reactivate Stripe subscription
      await this.stripe.subscriptions.update(subscription.externalId, {
        cancel_at_period_end: false,
      });

      // Update subscription record
      await this.subscriptionRepository.update(subscriptionId, {
        cancelAtPeriodEnd: false,
        status: SubscriptionStatus.ACTIVE,
      });

      this.logger.logBusiness('subscription_reactivated', userId, {
        subscriptionId,
      });

      return subscription;
    } catch (error) {
      this.logger.logError(error, 'SubscriptionsService.reactivateSubscription', {
        subscriptionId,
        userId,
      });
      throw error;
    }
  }

  async getSubscriptionStats(): Promise<{
    totalSubscriptions: number;
    activeSubscriptions: number;
    cancelledSubscriptions: number;
    subscriptionsByPlan: Record<string, number>;
  }> {
    const totalSubscriptions = await this.subscriptionRepository.countActiveSubscriptions();
    const activeSubscriptions = await this.subscriptionRepository.countActiveSubscriptions();
    const cancelledSubscriptions = await this.subscriptionRepository.countSubscriptionsByPlan(SubscriptionPlan.FREE);

    const subscriptionsByPlan = {
      [SubscriptionPlan.FREE]: await this.subscriptionRepository.countSubscriptionsByPlan(SubscriptionPlan.FREE),
      [SubscriptionPlan.BASIC]: await this.subscriptionRepository.countSubscriptionsByPlan(SubscriptionPlan.BASIC),
      [SubscriptionPlan.PREMIUM]: await this.subscriptionRepository.countSubscriptionsByPlan(SubscriptionPlan.PREMIUM),
      [SubscriptionPlan.CREATOR]: await this.subscriptionRepository.countSubscriptionsByPlan(SubscriptionPlan.CREATOR),
      [SubscriptionPlan.ENTERPRISE]: await this.subscriptionRepository.countSubscriptionsByPlan(SubscriptionPlan.ENTERPRISE),
    };

    return {
      totalSubscriptions,
      activeSubscriptions,
      cancelledSubscriptions,
      subscriptionsByPlan,
    };
  }

  async getExpiringSubscriptions(days: number = 7): Promise<Subscription[]> {
    return this.subscriptionRepository.findExpiringSubscriptions(days);
  }

  private getPlanDetails(plan: SubscriptionPlan): any {
    const plans = {
      [SubscriptionPlan.FREE]: {
        priceId: 'price_free',
        amount: 0,
        currency: 'usd',
        interval: 'month',
        intervalCount: 1,
        features: {
          maxVideos: 10,
          maxStorage: 1000, // MB
          analytics: false,
          monetization: false,
          prioritySupport: false,
        },
      },
      [SubscriptionPlan.BASIC]: {
        priceId: 'price_basic',
        amount: 999, // $9.99 in cents
        currency: 'usd',
        interval: 'month',
        intervalCount: 1,
        features: {
          maxVideos: 100,
          maxStorage: 10000, // MB
          analytics: true,
          monetization: false,
          prioritySupport: false,
        },
      },
      [SubscriptionPlan.PREMIUM]: {
        priceId: 'price_premium',
        amount: 1999, // $19.99 in cents
        currency: 'usd',
        interval: 'month',
        intervalCount: 1,
        features: {
          maxVideos: 1000,
          maxStorage: 100000, // MB
          analytics: true,
          monetization: true,
          prioritySupport: true,
        },
      },
      [SubscriptionPlan.CREATOR]: {
        priceId: 'price_creator',
        amount: 4999, // $49.99 in cents
        currency: 'usd',
        interval: 'month',
        intervalCount: 1,
        features: {
          maxVideos: -1, // Unlimited
          maxStorage: -1, // Unlimited
          analytics: true,
          monetization: true,
          prioritySupport: true,
          customBranding: true,
        },
      },
      [SubscriptionPlan.ENTERPRISE]: {
        priceId: 'price_enterprise',
        amount: 9999, // $99.99 in cents
        currency: 'usd',
        interval: 'month',
        intervalCount: 1,
        features: {
          maxVideos: -1, // Unlimited
          maxStorage: -1, // Unlimited
          analytics: true,
          monetization: true,
          prioritySupport: true,
          customBranding: true,
          apiAccess: true,
          whiteLabel: true,
        },
      },
    };

    return plans[plan];
  }
}
