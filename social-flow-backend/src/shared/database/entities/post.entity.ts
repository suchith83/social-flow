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

export enum PostType {
  TEXT = 'text',
  IMAGE = 'image',
  VIDEO = 'video',
  LINK = 'link',
  POLL = 'poll',
  THREAD = 'thread',
}

export enum PostStatus {
  DRAFT = 'draft',
  PUBLISHED = 'published',
  ARCHIVED = 'archived',
  DELETED = 'deleted',
}

@Entity('posts')
@Index(['userId'])
@Index(['type'])
@Index(['status'])
@Index(['createdAt'])
@Index(['likesCount'])
@Index(['commentsCount'])
@Index(['parentId'])
export class Post {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'text' })
  content: string;

  @Column({ default: PostType.TEXT })
  type: PostType;

  @Column({ default: PostStatus.PUBLISHED })
  status: PostStatus;

  @Column({ type: 'text', array: true, default: [] })
  hashtags: string[];

  @Column({ type: 'text', array: true, default: [] })
  mentions: string[];

  @Column({ type: 'text', array: true, default: [] })
  mediaUrls: string[];

  @Column({ type: 'jsonb', nullable: true })
  mediaMetadata: Record<string, any>[];

  @Column({ nullable: true })
  linkUrl: string;

  @Column({ nullable: true })
  linkTitle: string;

  @Column({ nullable: true })
  linkDescription: string;

  @Column({ nullable: true })
  linkImage: string;

  @Column({ type: 'jsonb', nullable: true })
  pollOptions: Record<string, any>[];

  @Column({ nullable: true })
  pollEndsAt: Date;

  @Column({ default: false })
  isPollClosed: boolean;

  @Column({ nullable: true })
  parentId: string;

  @Column({ nullable: true })
  threadId: string;

  @Column({ default: 0 })
  likesCount: number;

  @Column({ default: 0 })
  commentsCount: number;

  @Column({ default: 0 })
  sharesCount: number;

  @Column({ default: 0 })
  viewsCount: number;

  @Column({ default: 0 })
  retweetsCount: number;

  @Column({ default: false })
  isRetweet: boolean;

  @Column({ nullable: true })
  originalPostId: string;

  @Column({ default: false })
  isPinned: boolean;

  @Column({ default: false })
  isSensitive: boolean;

  @Column({ default: false })
  isAgeRestricted: boolean;

  @Column({ type: 'jsonb', nullable: true })
  location: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @Column({ nullable: true })
  scheduledAt: Date;

  @Column({ nullable: true })
  publishedAt: Date;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  // Relations
  @ManyToOne(() => User, (user) => user.posts, { onDelete: 'CASCADE' })
  user: User;

  @Column()
  userId: string;

  @OneToMany(() => Comment, (comment) => comment.post)
  comments: Comment[];

  @OneToMany(() => Like, (like) => like.post)
  likes: Like[];

  // Virtual fields
  get isPublished(): boolean {
    return this.status === PostStatus.PUBLISHED;
  }

  get isDraft(): boolean {
    return this.status === PostStatus.DRAFT;
  }

  get isArchived(): boolean {
    return this.status === PostStatus.ARCHIVED;
  }

  get isDeleted(): boolean {
    return this.status === PostStatus.DELETED;
  }

  get isThread(): boolean {
    return this.type === PostType.THREAD;
  }

  get isPoll(): boolean {
    return this.type === PostType.POLL;
  }

  get isPollActive(): boolean {
    return this.isPoll && !this.isPollClosed && (!this.pollEndsAt || new Date() < this.pollEndsAt);
  }

  get isRetweeted(): boolean {
    return this.isRetweet;
  }

  get hasMedia(): boolean {
    return this.mediaUrls.length > 0;
  }

  get hasLink(): boolean {
    return !!this.linkUrl;
  }

  get hasLocation(): boolean {
    return !!this.location;
  }

  get hasHashtags(): boolean {
    return this.hashtags.length > 0;
  }

  get hasMentions(): boolean {
    return this.mentions.length > 0;
  }

  get engagementRate(): number {
    const totalEngagements = this.likesCount + this.commentsCount + this.sharesCount;
    return this.viewsCount > 0 ? (totalEngagements / this.viewsCount) * 100 : 0;
  }

  get timeAgo(): string {
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - this.createdAt.getTime()) / 1000);
    
    if (diffInSeconds < 60) return `${diffInSeconds}s`;
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h`;
    if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 86400)}d`;
    if (diffInSeconds < 31536000) return `${Math.floor(diffInSeconds / 2592000)}mo`;
    return `${Math.floor(diffInSeconds / 31536000)}y`;
  }
}
