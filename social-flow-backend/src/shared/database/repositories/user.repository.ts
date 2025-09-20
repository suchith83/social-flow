import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, FindOneOptions } from 'typeorm';
import { User, UserRole, UserStatus } from '../entities/user.entity';
import { BaseRepository } from './base.repository';

@Injectable()
export class UserRepository extends BaseRepository<User> {
  constructor(
    @InjectRepository(User)
    private readonly userRepository: Repository<User>,
  ) {
    super(userRepository);
  }

  async findByEmail(email: string): Promise<User | null> {
    return this.userRepository.findOne({ where: { email } });
  }

  async findByUsername(username: string): Promise<User | null> {
    return this.userRepository.findOne({ where: { username } });
  }

  async findByEmailVerificationToken(token: string): Promise<User | null> {
    return this.userRepository.findOne({ where: { emailVerificationToken: token } });
  }

  async findByPasswordResetToken(token: string): Promise<User | null> {
    return this.userRepository.findOne({ 
      where: { passwordResetToken: token },
      // Check if token is not expired
    });
  }

  async findActiveUsers(limit: number = 10, offset: number = 0): Promise<User[]> {
    return this.userRepository.find({
      where: { status: UserStatus.ACTIVE },
      take: limit,
      skip: offset,
      order: { lastActiveAt: 'DESC' },
    });
  }

  async findUsersByRole(role: UserRole, limit: number = 10, offset: number = 0): Promise<User[]> {
    return this.userRepository.find({
      where: { role },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async findUsersByStatus(status: UserStatus, limit: number = 10, offset: number = 0): Promise<User[]> {
    return this.userRepository.find({
      where: { status },
      take: limit,
      skip: offset,
      order: { createdAt: 'DESC' },
    });
  }

  async searchUsers(query: string, limit: number = 10, offset: number = 0): Promise<User[]> {
    return this.userRepository
      .createQueryBuilder('user')
      .where('user.username ILIKE :query OR user.firstName ILIKE :query OR user.lastName ILIKE :query', {
        query: `%${query}%`,
      })
      .andWhere('user.status = :status', { status: UserStatus.ACTIVE })
      .take(limit)
      .skip(offset)
      .orderBy('user.createdAt', 'DESC')
      .getMany();
  }

  async findTopCreators(limit: number = 10): Promise<User[]> {
    return this.userRepository
      .createQueryBuilder('user')
      .where('user.role IN (:...roles)', { roles: [UserRole.CREATOR, UserRole.ADMIN] })
      .andWhere('user.status = :status', { status: UserStatus.ACTIVE })
      .orderBy('user.totalViews', 'DESC')
      .take(limit)
      .getMany();
  }

  async findUsersWithMostFollowers(limit: number = 10): Promise<User[]> {
    return this.userRepository
      .createQueryBuilder('user')
      .where('user.status = :status', { status: UserStatus.ACTIVE })
      .orderBy('user.followersCount', 'DESC')
      .take(limit)
      .getMany();
  }

  async updateLastActive(userId: string): Promise<void> {
    await this.userRepository.update(userId, { lastActiveAt: new Date() });
  }

  async updatePassword(userId: string, hashedPassword: string): Promise<void> {
    await this.userRepository.update(userId, { 
      password: hashedPassword,
      passwordResetToken: null,
      passwordResetExpires: null,
    });
  }

  async updateEmailVerification(userId: string, verified: boolean): Promise<void> {
    await this.userRepository.update(userId, { 
      emailVerified: verified,
      emailVerificationToken: verified ? null : undefined,
    });
  }

  async updatePasswordResetToken(userId: string, token: string, expiresAt: Date): Promise<void> {
    await this.userRepository.update(userId, {
      passwordResetToken: token,
      passwordResetExpires: expiresAt,
    });
  }

  async incrementFollowersCount(userId: string): Promise<void> {
    await this.userRepository.increment({ id: userId }, 'followersCount', 1);
  }

  async decrementFollowersCount(userId: string): Promise<void> {
    await this.userRepository.decrement({ id: userId }, 'followersCount', 1);
  }

  async incrementFollowingCount(userId: string): Promise<void> {
    await this.userRepository.increment({ id: userId }, 'followingCount', 1);
  }

  async decrementFollowingCount(userId: string): Promise<void> {
    await this.userRepository.decrement({ id: userId }, 'followingCount', 1);
  }

  async incrementPostsCount(userId: string): Promise<void> {
    await this.userRepository.increment({ id: userId }, 'postsCount', 1);
  }

  async decrementPostsCount(userId: string): Promise<void> {
    await this.userRepository.decrement({ id: userId }, 'postsCount', 1);
  }

  async incrementVideosCount(userId: string): Promise<void> {
    await this.userRepository.increment({ id: userId }, 'videosCount', 1);
  }

  async decrementVideosCount(userId: string): Promise<void> {
    await this.userRepository.decrement({ id: userId }, 'videosCount', 1);
  }

  async incrementTotalViews(userId: string, views: number = 1): Promise<void> {
    await this.userRepository.increment({ id: userId }, 'totalViews', views);
  }

  async incrementTotalLikes(userId: string, likes: number = 1): Promise<void> {
    await this.userRepository.increment({ id: userId }, 'totalLikes', likes);
  }

  async updateUserStats(userId: string, stats: Partial<{
    followersCount: number;
    followingCount: number;
    postsCount: number;
    videosCount: number;
    totalViews: number;
    totalLikes: number;
  }>): Promise<void> {
    await this.userRepository.update(userId, stats);
  }

  async findUsersByIds(userIds: string[]): Promise<User[]> {
    return this.userRepository.findByIds(userIds);
  }

  async countUsersByRole(role: UserRole): Promise<number> {
    return this.userRepository.count({ where: { role } });
  }

  async countUsersByStatus(status: UserStatus): Promise<number> {
    return this.userRepository.count({ where: { status } });
  }

  async countActiveUsers(): Promise<number> {
    return this.userRepository.count({ where: { status: UserStatus.ACTIVE } });
  }

  async countVerifiedUsers(): Promise<number> {
    return this.userRepository.count({ where: { emailVerified: true } });
  }
}
