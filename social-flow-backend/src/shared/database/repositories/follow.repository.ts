import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Follow } from '../entities/follow.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class FollowRepository extends BaseRepository<Follow> {
  constructor(
    @InjectRepository(Follow)
    private readonly followRepository: Repository<Follow>,
  ) {
    super(followRepository);
  }

  async findByFollowerAndFollowing(followerId: string, followingId: string): Promise<Follow | null> {
    return this.followRepository.findOne({ where: { followerId, followingId } });
  }

  async findFollowers(userId: string, limit: number = 10, offset: number = 0): Promise<Follow[]> {
    return this.followRepository.find({
      where: { followingId: userId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
      relations: ['follower'],
    });
  }

  async findFollowing(userId: string, limit: number = 10, offset: number = 0): Promise<Follow[]> {
    return this.followRepository.find({
      where: { followerId: userId },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
      relations: ['following'],
    });
  }

  async countFollowers(userId: string): Promise<number> {
    return this.followRepository.count({ where: { followingId: userId } });
  }

  async countFollowing(userId: string): Promise<number> {
    return this.followRepository.count({ where: { followerId: userId } });
  }

  async isFollowing(followerId: string, followingId: string): Promise<boolean> {
    const follow = await this.findByFollowerAndFollowing(followerId, followingId);
    return !!follow;
  }
}
