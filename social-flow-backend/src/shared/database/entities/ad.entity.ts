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

export enum AdType {
  BANNER = 'banner',
  VIDEO = 'video',
  SPONSORED_POST = 'sponsored_post',
  NATIVE = 'native',
  INTERSTITIAL = 'interstitial',
}

export enum AdStatus {
  DRAFT = 'draft',
  PENDING = 'pending',
  APPROVED = 'approved',
  REJECTED = 'rejected',
  ACTIVE = 'active',
  PAUSED = 'paused',
  COMPLETED = 'completed',
}

export enum AdTargeting {
  ALL = 'all',
  AGE = 'age',
  LOCATION = 'location',
  INTERESTS = 'interests',
  BEHAVIOR = 'behavior',
  DEMOGRAPHICS = 'demographics',
}

@Entity('ads')
@Index(['advertiserId'])
@Index(['type'])
@Index(['status'])
@Index(['targeting'])
@Index(['createdAt'])
@Index(['startDate'])
@Index(['endDate'])
export class Ad {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  title: string;

  @Column({ type: 'text' })
  description: string;

  @Column({ default: AdType.BANNER })
  type: AdType;

  @Column({ default: AdStatus.DRAFT })
  status: AdStatus;

  @Column({ type: 'text', array: true, default: [] })
  mediaUrls: string[];

  @Column({ nullable: true })
  clickUrl: string;

  @Column({ default: 0 })
  budget: number; // in cents

  @Column({ default: 0 })
  spent: number; // in cents

  @Column({ default: 0 })
  bidAmount: number; // in cents

  @Column({ default: 0 })
  impressions: number;

  @Column({ default: 0 })
  clicks: number;

  @Column({ default: 0 })
  conversions: number;

  @Column({ default: 0 })
  ctr: number; // click-through rate

  @Column({ default: 0 })
  cpm: number; // cost per mille

  @Column({ default: 0 })
  cpc: number; // cost per click

  @Column({ default: 0 })
  cpa: number; // cost per acquisition

  @Column({ default: AdTargeting.ALL })
  targeting: AdTargeting;

  @Column({ type: 'jsonb', nullable: true })
  targetingCriteria: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  demographics: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  interests: string[];

  @Column({ type: 'jsonb', nullable: true })
  locations: string[];

  @Column({ type: 'jsonb', nullable: true })
  behaviors: string[];

  @Column({ nullable: true })
  startDate: Date;

  @Column({ nullable: true })
  endDate: Date;

  @Column({ default: false })
  isActive: boolean;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  // Relations
  @ManyToOne(() => User, { onDelete: 'CASCADE' })
  advertiser: User;

  @Column()
  advertiserId: string;

  // Virtual fields
  get isActive(): boolean {
    const now = new Date();
    return this.status === AdStatus.ACTIVE && 
           (!this.startDate || this.startDate <= now) && 
           (!this.endDate || this.endDate >= now);
  }

  get remainingBudget(): number {
    return this.budget - this.spent;
  }

  get budgetUtilization(): number {
    return this.budget > 0 ? (this.spent / this.budget) * 100 : 0;
  }

  get performanceScore(): number {
    if (this.impressions === 0) return 0;
    return (this.ctr * 0.4 + (this.conversions / this.clicks) * 0.6) * 100;
  }
}
