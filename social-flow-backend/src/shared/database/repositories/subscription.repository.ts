import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Subscription, SubscriptionStatus, SubscriptionPlan } from '../entities/subscription.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class SubscriptionRepository extends BaseRepository<Subscription> {
  constructor(
    @InjectRepository(Subscription)
    private readonly subscriptionRepository: Repository<Subscription>,
  ) {
    super(subscriptionRepository);
  }

  async findByUser(userId: string): Promise<Subscription | null> {
    return this.subscriptionRepository.findOne({ where: { userId } });
  }

  async findByStatus(status: SubscriptionStatus, limit: number = 10, offset: number = 0): Promise<Subscription[]> {
    return this.subscriptionRepository.find({
      where: { status },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByPlan(plan: SubscriptionPlan, limit: number = 10, offset: number = 0): Promise<Subscription[]> {
    return this.subscriptionRepository.find({
      where: { plan },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByExternalId(externalId: string): Promise<Subscription | null> {
    return this.subscriptionRepository.findOne({ where: { externalId } });
  }

  async updateStatus(subscriptionId: string, status: SubscriptionStatus): Promise<void> {
    await this.subscriptionRepository.update(subscriptionId, { status });
  }

  async findExpiringSubscriptions(days: number = 7): Promise<Subscription[]> {
    const futureDate = new Date();
    futureDate.setDate(futureDate.getDate() + days);

    return this.subscriptionRepository
      .createQueryBuilder('subscription')
      .where('subscription.status = :status', { status: SubscriptionStatus.ACTIVE })
      .andWhere('subscription.currentPeriodEnd <= :futureDate', { futureDate })
      .orderBy('subscription.currentPeriodEnd', 'ASC')
      .getMany();
  }

  async countActiveSubscriptions(): Promise<number> {
    return this.subscriptionRepository.count({ where: { status: SubscriptionStatus.ACTIVE } });
  }

  async countSubscriptionsByPlan(plan: SubscriptionPlan): Promise<number> {
    return this.subscriptionRepository.count({ where: { plan, status: SubscriptionStatus.ACTIVE } });
  }
}
