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

export enum SubscriptionStatus {
  ACTIVE = 'active',
  CANCELLED = 'cancelled',
  PAST_DUE = 'past_due',
  UNPAID = 'unpaid',
  INCOMPLETE = 'incomplete',
  INCOMPLETE_EXPIRED = 'incomplete_expired',
  TRIALING = 'trialing',
}

export enum SubscriptionPlan {
  FREE = 'free',
  BASIC = 'basic',
  PREMIUM = 'premium',
  CREATOR = 'creator',
  ENTERPRISE = 'enterprise',
}

@Entity('subscriptions')
@Index(['userId'])
@Index(['status'])
@Index(['plan'])
@Index(['createdAt'])
@Index(['currentPeriodStart'])
@Index(['currentPeriodEnd'])
export class Subscription {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ default: SubscriptionPlan.FREE })
  plan: SubscriptionPlan;

  @Column({ default: SubscriptionStatus.ACTIVE })
  status: SubscriptionStatus;

  @Column({ nullable: true })
  externalId: string; // Stripe subscription ID

  @Column({ nullable: true })
  priceId: string; // Stripe price ID

  @Column({ default: 0 })
  amount: number; // in cents

  @Column({ default: 'usd' })
  currency: string;

  @Column({ default: 'month' })
  interval: string; // month, year

  @Column({ default: 1 })
  intervalCount: number;

  @Column({ nullable: true })
  trialStart: Date;

  @Column({ nullable: true })
  trialEnd: Date;

  @Column({ nullable: true })
  currentPeriodStart: Date;

  @Column({ nullable: true })
  currentPeriodEnd: Date;

  @Column({ nullable: true })
  cancelAtPeriodEnd: boolean;

  @Column({ nullable: true })
  cancelledAt: Date;

  @Column({ nullable: true })
  endedAt: Date;

  @Column({ type: 'jsonb', nullable: true })
  features: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  // Relations
  @ManyToOne(() => User, (user) => user.subscriptions, { onDelete: 'CASCADE' })
  user: User;

  @Column()
  userId: string;

  // Virtual fields
  get isActive(): boolean {
    return this.status === SubscriptionStatus.ACTIVE;
  }

  get isCancelled(): boolean {
    return this.status === SubscriptionStatus.CANCELLED;
  }

  get isTrialing(): boolean {
    return this.status === SubscriptionStatus.TRIALING;
  }

  get isPastDue(): boolean {
    return this.status === SubscriptionStatus.PAST_DUE;
  }

  get isUnpaid(): boolean {
    return this.status === SubscriptionStatus.UNPAID;
  }

  get amountFormatted(): string {
    return `${(this.amount / 100).toFixed(2)} ${this.currency.toUpperCase()}`;
  }

  get isInTrial(): boolean {
    if (!this.trialEnd) return false;
    return new Date() < this.trialEnd;
  }

  get daysUntilRenewal(): number {
    if (!this.currentPeriodEnd) return 0;
    const now = new Date();
    const diffTime = this.currentPeriodEnd.getTime() - now.getTime();
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  }
}
