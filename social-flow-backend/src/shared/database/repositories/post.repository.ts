import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Post, PostType, PostStatus } from '../entities/post.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class PostRepository extends BaseRepository<Post> {
  constructor(
    @InjectRepository(Post)
    private readonly postRepository: Repository<Post>,
  ) {
    super(postRepository);
  }

  async findByUserId(userId: string, limit: number = 10, offset: number = 0): Promise<Post[]> {
    return this.postRepository.find({
      where: { userId, status: PostStatus.PUBLISHED },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findFeedPosts(userIds: string[], limit: number = 10, offset: number = 0): Promise<Post[]> {
    return this.postRepository
      .createQueryBuilder('post')
      .where('post.userId IN (:...userIds)', { userIds })
      .andWhere('post.status = :status', { status: PostStatus.PUBLISHED })
      .orderBy('post.createdAt', 'DESC')
      .take(limit)
      .skip(offset)
      .getMany();
  }

  async findPostsByHashtag(hashtag: string, limit: number = 10, offset: number = 0): Promise<Post[]> {
    return this.postRepository
      .createQueryBuilder('post')
      .where('post.status = :status', { status: PostStatus.PUBLISHED })
      .andWhere(':hashtag = ANY(post.hashtags)', { hashtag })
      .orderBy('post.createdAt', 'DESC')
      .take(limit)
      .skip(offset)
      .getMany();
  }

  async searchPosts(query: string, limit: number = 10, offset: number = 0): Promise<Post[]> {
    return this.postRepository
      .createQueryBuilder('post')
      .where('post.status = :status', { status: PostStatus.PUBLISHED })
      .andWhere('post.content ILIKE :query', { query: `%${query}%` })
      .orderBy('post.createdAt', 'DESC')
      .take(limit)
      .skip(offset)
      .getMany();
  }

  async incrementLikesCount(postId: string): Promise<void> {
    await this.postRepository.increment({ id: postId }, 'likesCount', 1);
  }

  async decrementLikesCount(postId: string): Promise<void> {
    await this.postRepository.decrement({ id: postId }, 'likesCount', 1);
  }

  async incrementCommentsCount(postId: string): Promise<void> {
    await this.postRepository.increment({ id: postId }, 'commentsCount', 1);
  }

  async decrementCommentsCount(postId: string): Promise<void> {
    await this.postRepository.decrement({ id: postId }, 'commentsCount', 1);
  }

  async incrementSharesCount(postId: string): Promise<void> {
    await this.postRepository.increment({ id: postId }, 'sharesCount', 1);
  }

  async incrementViewsCount(postId: string): Promise<void> {
    await this.postRepository.increment({ id: postId }, 'viewsCount', 1);
  }
}
