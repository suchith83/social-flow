import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Like, LikeType } from '../entities/like.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class LikeRepository extends BaseRepository<Like> {
  constructor(
    @InjectRepository(Like)
    private readonly likeRepository: Repository<Like>,
  ) {
    super(likeRepository);
  }

  async findByUserAndPost(userId: string, postId: string): Promise<Like | null> {
    return this.likeRepository.findOne({ where: { userId, postId } });
  }

  async findByUserAndVideo(userId: string, videoId: string): Promise<Like | null> {
    return this.likeRepository.findOne({ where: { userId, videoId } });
  }

  async findByUserAndComment(userId: string, commentId: string): Promise<Like | null> {
    return this.likeRepository.findOne({ where: { userId, commentId } });
  }

  async findLikesByPost(postId: string, limit: number = 10, offset: number = 0): Promise<Like[]> {
    return this.likeRepository.find({
      where: { postId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
      relations: ['user'],
    });
  }

  async findLikesByVideo(videoId: string, limit: number = 10, offset: number = 0): Promise<Like[]> {
    return this.likeRepository.find({
      where: { videoId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
      relations: ['user'],
    });
  }

  async findLikesByComment(commentId: string, limit: number = 10, offset: number = 0): Promise<Like[]> {
    return this.likeRepository.find({
      where: { commentId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
      relations: ['user'],
    });
  }

  async countLikesByPost(postId: string): Promise<number> {
    return this.likeRepository.count({ where: { postId } });
  }

  async countLikesByVideo(videoId: string): Promise<number> {
    return this.likeRepository.count({ where: { videoId } });
  }

  async countLikesByComment(commentId: string): Promise<number> {
    return this.likeRepository.count({ where: { commentId } });
  }

  async countLikesByUser(userId: string): Promise<number> {
    return this.likeRepository.count({ where: { userId } });
  }
}
