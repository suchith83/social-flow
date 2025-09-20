import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  Index,
  Unique,
} from 'typeorm';
import { Video } from './video.entity';

@Entity('view_counts')
@Index(['videoId'])
@Index(['date'])
@Index(['createdAt'])
@Unique(['videoId', 'date'])
export class ViewCount {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ default: 0 })
  count: number;

  @Column({ default: 0 })
  uniqueViews: number;

  @Column({ default: 0 })
  watchTime: number; // in seconds

  @Column({ default: 0 })
  averageWatchTime: number; // in seconds

  @Column({ default: 0 })
  retentionRate: number; // percentage

  @Column({ type: 'date' })
  date: Date;

  @Column({ type: 'jsonb', nullable: true })
  hourlyBreakdown: Record<string, number>;

  @Column({ type: 'jsonb', nullable: true })
  countryBreakdown: Record<string, number>;

  @Column({ type: 'jsonb', nullable: true })
  deviceBreakdown: Record<string, number>;

  @Column({ type: 'jsonb', nullable: true })
  sourceBreakdown: Record<string, number>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  // Relations
  @ManyToOne(() => Video, (video) => video.viewCounts, { onDelete: 'CASCADE' })
  video: Video;

  @Column()
  videoId: string;
}
