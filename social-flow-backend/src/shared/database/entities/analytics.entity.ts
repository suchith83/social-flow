import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  Index,
} from 'typeorm';

export enum AnalyticsType {
  PAGE_VIEW = 'page_view',
  VIDEO_VIEW = 'video_view',
  POST_VIEW = 'post_view',
  USER_ACTION = 'user_action',
  SYSTEM_EVENT = 'system_event',
  PERFORMANCE = 'performance',
  ERROR = 'error',
}

export enum AnalyticsCategory {
  USER = 'user',
  CONTENT = 'content',
  ENGAGEMENT = 'engagement',
  MONETIZATION = 'monetization',
  TECHNICAL = 'technical',
  BUSINESS = 'business',
}

@Entity('analytics')
@Index(['type'])
@Index(['category'])
@Index(['userId'])
@Index(['entityType'])
@Index(['entityId'])
@Index(['createdAt'])
@Index(['date'])
export class Analytics {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  type: AnalyticsType;

  @Column()
  category: AnalyticsCategory;

  @Column({ nullable: true })
  userId: string;

  @Column({ nullable: true })
  sessionId: string;

  @Column({ nullable: true })
  entityType: string; // 'video', 'post', 'user', etc.

  @Column({ nullable: true })
  entityId: string;

  @Column()
  event: string;

  @Column({ type: 'jsonb', nullable: true })
  properties: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  context: Record<string, any>;

  @Column({ nullable: true })
  ipAddress: string;

  @Column({ nullable: true })
  userAgent: string;

  @Column({ nullable: true })
  referrer: string;

  @Column({ nullable: true })
  country: string;

  @Column({ nullable: true })
  city: string;

  @Column({ nullable: true })
  device: string;

  @Column({ nullable: true })
  browser: string;

  @Column({ nullable: true })
  os: string;

  @Column({ default: 0 })
  value: number;

  @Column({ type: 'date' })
  date: Date;

  @Column({ type: 'timestamp' })
  timestamp: Date;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}
