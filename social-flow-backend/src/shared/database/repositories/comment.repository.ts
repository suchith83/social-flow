import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Comment } from '../entities/comment.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class CommentRepository extends BaseRepository<Comment> {
  constructor(
    @InjectRepository(Comment)
    private readonly commentRepository: Repository<Comment>,
  ) {
    super(commentRepository);
  }

  async findByPostId(postId: string, limit: number = 10, offset: number = 0): Promise<Comment[]> {
    return this.commentRepository.find({
      where: { postId, parentId: null },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
      relations: ['user', 'replies'],
    });
  }

  async findByVideoId(videoId: string, limit: number = 10, offset: number = 0): Promise<Comment[]> {
    return this.commentRepository.find({
      where: { videoId, parentId: null },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
      relations: ['user', 'replies'],
    });
  }

  async findReplies(parentId: string, limit: number = 10, offset: number = 0): Promise<Comment[]> {
    return this.commentRepository.find({
      where: { parentId },
      take: limit,
      skip: offset,
      order: { createdAt: 'ASC' },
      relations: ['user'],
    });
  }

  async incrementLikesCount(commentId: string): Promise<void> {
    await this.commentRepository.increment({ id: commentId }, 'likesCount', 1);
  }

  async decrementLikesCount(commentId: string): Promise<void> {
    await this.commentRepository.decrement({ id: commentId }, 'likesCount', 1);
  }

  async incrementRepliesCount(commentId: string): Promise<void> {
    await this.commentRepository.increment({ id: commentId }, 'repliesCount', 1);
  }

  async decrementRepliesCount(commentId: string): Promise<void> {
    await this.commentRepository.decrement({ id: commentId }, 'repliesCount', 1);
  }
}
