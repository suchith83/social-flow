import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  OneToMany,
  Index,
} from 'typeorm';
import { User } from './user.entity';
import { Comment } from './comment.entity';
import { Like } from './like.entity';
import { ViewCount } from './view-count.entity';

export enum VideoStatus {
  UPLOADING = 'uploading',
  PROCESSING = 'processing',
  PROCESSED = 'processed',
  FAILED = 'failed',
  PRIVATE = 'private',
  UNLISTED = 'unlisted',
  PUBLIC = 'public',
}

export enum VideoVisibility {
  PUBLIC = 'public',
  UNLISTED = 'unlisted',
  PRIVATE = 'private',
}

@Entity('videos')
@Index(['userId'])
@Index(['status'])
@Index(['visibility'])
@Index(['createdAt'])
@Index(['viewsCount'])
@Index(['likesCount'])
export class Video {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  title: string;

  @Column({ type: 'text', nullable: true })
  description: string;

  @Column({ type: 'text', array: true, default: [] })
  tags: string[];

  @Column({ nullable: true })
  thumbnail: string;

  @Column({ nullable: true })
  thumbnailLarge: string;

  @Column({ nullable: true })
  videoUrl: string;

  @Column({ nullable: true })
  hlsUrl: string;

  @Column({ nullable: true })
  dashUrl: string;

  @Column({ nullable: true })
  previewUrl: string;

  @Column({ type: 'jsonb', nullable: true })
  resolutions: Record<string, string>;

  @Column({ type: 'jsonb', nullable: true })
  thumbnails: Record<string, string>;

  @Column({ nullable: true })
  duration: number; // in seconds

  @Column({ nullable: true })
  fileSize: number; // in bytes

  @Column({ nullable: true })
  width: number;

  @Column({ nullable: true })
  height: number;

  @Column({ nullable: true })
  fps: number;

  @Column({ nullable: true })
  bitrate: number;

  @Column({ nullable: true })
  codec: string;

  @Column({ default: VideoStatus.UPLOADING })
  status: VideoStatus;

  @Column({ default: VideoVisibility.PUBLIC })
  visibility: VideoVisibility;

  @Column({ default: false })
  isLive: boolean;

  @Column({ nullable: true })
  liveStreamUrl: string;

  @Column({ nullable: true })
  liveStreamKey: string;

  @Column({ default: 0 })
  viewsCount: number;

  @Column({ default: 0 })
  likesCount: number;

  @Column({ default: 0 })
  commentsCount: number;

  @Column({ default: 0 })
  sharesCount: number;

  @Column({ default: 0 })
  watchTime: number; // total watch time in seconds

  @Column({ default: 0 })
  averageWatchTime: number; // average watch time in seconds

  @Column({ default: 0 })
  retentionRate: number; // percentage

  @Column({ default: false })
  isMonetized: boolean;

  @Column({ default: false })
  isAgeRestricted: boolean;

  @Column({ default: false })
  isCopyrighted: boolean;

  @Column({ nullable: true })
  copyrightInfo: string;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  processingMetadata: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  analytics: Record<string, any>;

  @Column({ nullable: true })
  scheduledAt: Date;

  @Column({ nullable: true })
  publishedAt: Date;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  // Relations
  @ManyToOne(() => User, (user) => user.videos, { onDelete: 'CASCADE' })
  user: User;

  @Column()
  userId: string;

  @OneToMany(() => Comment, (comment) => comment.video)
  comments: Comment[];

  @OneToMany(() => Like, (like) => like.video)
  likes: Like[];

  @OneToMany(() => ViewCount, (viewCount) => viewCount.video)
  viewCounts: ViewCount[];

  // Virtual fields
  get isPublic(): boolean {
    return this.visibility === VideoVisibility.PUBLIC;
  }

  get isPrivate(): boolean {
    return this.visibility === VideoVisibility.PRIVATE;
  }

  get isUnlisted(): boolean {
    return this.visibility === VideoVisibility.UNLISTED;
  }

  get isProcessed(): boolean {
    return this.status === VideoStatus.PROCESSED;
  }

  get isProcessing(): boolean {
    return this.status === VideoStatus.PROCESSING;
  }

  get isUploading(): boolean {
    return this.status === VideoStatus.UPLOADING;
  }

  get isFailed(): boolean {
    return this.status === VideoStatus.FAILED;
  }

  get durationFormatted(): string {
    if (!this.duration) return '0:00';
    const minutes = Math.floor(this.duration / 60);
    const seconds = Math.floor(this.duration % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  get fileSizeFormatted(): string {
    if (!this.fileSize) return '0 B';
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(this.fileSize) / Math.log(1024));
    return `${(this.fileSize / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
  }

  get aspectRatio(): number {
    if (!this.width || !this.height) return 16 / 9;
    return this.width / this.height;
  }

  get isLandscape(): boolean {
    return this.aspectRatio > 1;
  }

  get isPortrait(): boolean {
    return this.aspectRatio < 1;
  }

  get isSquare(): boolean {
    return Math.abs(this.aspectRatio - 1) < 0.1;
  }
}
