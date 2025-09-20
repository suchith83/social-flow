import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  ManyToOne,
  Index,
  Unique,
} from 'typeorm';
import { User } from './user.entity';
import { Post } from './post.entity';
import { Video } from './video.entity';
import { Comment } from './comment.entity';

export enum LikeType {
  LIKE = 'like',
  LOVE = 'love',
  LAUGH = 'laugh',
  ANGRY = 'angry',
  SAD = 'sad',
  WOW = 'wow',
}

@Entity('likes')
@Index(['userId'])
@Index(['postId'])
@Index(['videoId'])
@Index(['commentId'])
@Index(['createdAt'])
@Unique(['userId', 'postId'])
@Unique(['userId', 'videoId'])
@Unique(['userId', 'commentId'])
export class Like {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ default: LikeType.LIKE })
  type: LikeType;

  @CreateDateColumn()
  createdAt: Date;

  // Relations
  @ManyToOne(() => User, (user) => user.likes, { onDelete: 'CASCADE' })
  user: User;

  @Column()
  userId: string;

  @ManyToOne(() => Post, (post) => post.likes, { onDelete: 'CASCADE', nullable: true })
  post: Post;

  @Column({ nullable: true })
  postId: string;

  @ManyToOne(() => Video, (video) => video.likes, { onDelete: 'CASCADE', nullable: true })
  video: Video;

  @Column({ nullable: true })
  videoId: string;

  @ManyToOne(() => Comment, (comment) => comment.likes, { onDelete: 'CASCADE', nullable: true })
  comment: Comment;

  @Column({ nullable: true })
  commentId: string;

  // Virtual fields
  get isPostLike(): boolean {
    return !!this.postId;
  }

  get isVideoLike(): boolean {
    return !!this.videoId;
  }

  get isCommentLike(): boolean {
    return !!this.commentId;
  }
}
