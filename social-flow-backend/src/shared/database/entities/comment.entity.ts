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
import { Post } from './post.entity';
import { Video } from './video.entity';
import { Like } from './like.entity';

@Entity('comments')
@Index(['userId'])
@Index(['postId'])
@Index(['videoId'])
@Index(['parentId'])
@Index(['createdAt'])
@Index(['likesCount'])
export class Comment {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'text' })
  content: string;

  @Column({ nullable: true })
  parentId: string;

  @Column({ default: 0 })
  likesCount: number;

  @Column({ default: 0 })
  repliesCount: number;

  @Column({ default: false })
  isEdited: boolean;

  @Column({ nullable: true })
  editedAt: Date;

  @Column({ default: false })
  isPinned: boolean;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  // Relations
  @ManyToOne(() => User, (user) => user.comments, { onDelete: 'CASCADE' })
  user: User;

  @Column()
  userId: string;

  @ManyToOne(() => Post, (post) => post.comments, { onDelete: 'CASCADE', nullable: true })
  post: Post;

  @Column({ nullable: true })
  postId: string;

  @ManyToOne(() => Video, (video) => video.comments, { onDelete: 'CASCADE', nullable: true })
  video: Video;

  @Column({ nullable: true })
  videoId: string;

  @OneToMany(() => Like, (like) => like.comment)
  likes: Like[];

  @OneToMany(() => Comment, (comment) => comment.parent)
  replies: Comment[];

  @ManyToOne(() => Comment, (comment) => comment.replies, { nullable: true })
  parent: Comment;

  // Virtual fields
  get isReply(): boolean {
    return !!this.parentId;
  }

  get isTopLevel(): boolean {
    return !this.parentId;
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
