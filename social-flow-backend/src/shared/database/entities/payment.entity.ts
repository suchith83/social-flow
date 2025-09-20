import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  Index,
} from 'typeorm';
import { User } from './user.entity';

export enum PaymentStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  REFUNDED = 'refunded',
}

export enum PaymentMethod {
  STRIPE = 'stripe',
  PAYPAL = 'paypal',
  APPLE_PAY = 'apple_pay',
  GOOGLE_PAY = 'google_pay',
  BANK_TRANSFER = 'bank_transfer',
  CRYPTO = 'crypto',
}

export enum PaymentType {
  SUBSCRIPTION = 'subscription',
  DONATION = 'donation',
  PURCHASE = 'purchase',
  REFUND = 'refund',
  PAYOUT = 'payout',
}

@Entity('payments')
@Index(['userId'])
@Index(['status'])
@Index(['method'])
@Index(['type'])
@Index(['createdAt'])
export class Payment {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  amount: number; // in cents

  @Column()
  currency: string;

  @Column({ default: PaymentStatus.PENDING })
  status: PaymentStatus;

  @Column({ default: PaymentMethod.STRIPE })
  method: PaymentMethod;

  @Column({ default: PaymentType.PURCHASE })
  type: PaymentType;

  @Column({ nullable: true })
  externalId: string; // Stripe payment intent ID, PayPal transaction ID, etc.

  @Column({ nullable: true })
  description: string;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  paymentData: Record<string, any>;

  @Column({ nullable: true })
  failureReason: string;

  @Column({ nullable: true })
  refundId: string;

  @Column({ nullable: true })
  refundAmount: number;

  @Column({ nullable: true })
  refundReason: string;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  // Relations
  @ManyToOne(() => User, (user) => user.payments, { onDelete: 'CASCADE' })
  user: User;

  @Column()
  userId: string;

  // Virtual fields
  get amountFormatted(): string {
    return `${(this.amount / 100).toFixed(2)} ${this.currency.toUpperCase()}`;
  }

  get isCompleted(): boolean {
    return this.status === PaymentStatus.COMPLETED;
  }

  get isPending(): boolean {
    return this.status === PaymentStatus.PENDING;
  }

  get isFailed(): boolean {
    return this.status === PaymentStatus.FAILED;
  }

  get isRefunded(): boolean {
    return this.status === PaymentStatus.REFUNDED;
  }
}
