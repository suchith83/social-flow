import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Payment, PaymentStatus, PaymentType } from '../entities/payment.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class PaymentRepository extends BaseRepository<Payment> {
  constructor(
    @InjectRepository(Payment)
    private readonly paymentRepository: Repository<Payment>,
  ) {
    super(paymentRepository);
  }

  async findByUser(userId: string, limit: number = 10, offset: number = 0): Promise<Payment[]> {
    return this.paymentRepository.find({
      where: { userId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByStatus(status: PaymentStatus, limit: number = 10, offset: number = 0): Promise<Payment[]> {
    return this.paymentRepository.find({
      where: { status },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByType(type: PaymentType, limit: number = 10, offset: number = 0): Promise<Payment[]> {
    return this.paymentRepository.find({
      where: { type },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findByExternalId(externalId: string): Promise<Payment | null> {
    return this.paymentRepository.findOne({ where: { externalId } });
  }

  async updateStatus(paymentId: string, status: PaymentStatus, failureReason?: string): Promise<void> {
    const updateData: Partial<Payment> = { status };
    if (failureReason) {
      updateData.failureReason = failureReason;
    }
    await this.paymentRepository.update(paymentId, updateData);
  }

  async getTotalRevenue(userId?: string, startDate?: Date, endDate?: Date): Promise<number> {
    const query = this.paymentRepository
      .createQueryBuilder('payment')
      .select('SUM(payment.amount)', 'total')
      .where('payment.status = :status', { status: PaymentStatus.COMPLETED });

    if (userId) {
      query.andWhere('payment.userId = :userId', { userId });
    }

    if (startDate) {
      query.andWhere('payment.createdAt >= :startDate', { startDate });
    }

    if (endDate) {
      query.andWhere('payment.createdAt <= :endDate', { endDate });
    }

    const result = await query.getRawOne();
    return parseInt(result.total) || 0;
  }
}
