import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { FollowRepository } from '../shared/database/repositories/follow.repository';
import { User, UserRole } from '../shared/database/entities/user.entity';
import { S3Service } from '../shared/aws/s3.service';
import { LoggerService } from '../shared/logger/logger.service';

export interface UpdateProfileRequest {
  firstName?: string;
  lastName?: string;
  bio?: string;
  website?: string;
  location?: string;
  birthDate?: Date;
  preferences?: Record<string, any>;
  socialLinks?: Record<string, string>;
}

export interface FollowUserRequest {
  userId: string;
  targetUserId: string;
}

@Injectable()
export class UsersService {
  constructor(
    private userRepository: UserRepository,
    private followRepository: FollowRepository,
    private s3Service: S3Service,
    private logger: LoggerService,
  ) {}

  async getProfile(userId: string): Promise<User> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    return user;
  }

  async updateProfile(userId: string, updateData: UpdateProfileRequest): Promise<User> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    const updatedUser = await this.userRepository.update(userId, updateData);
    
    this.logger.logBusiness('profile_updated', userId, updateData);
    
    return updatedUser;
  }

  async uploadAvatar(userId: string, file: Express.Multer.File): Promise<string> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Upload to S3
    const avatarUrl = await this.s3Service.uploadUserAvatar(userId, file.buffer, file.originalname);

    // Update user avatar
    await this.userRepository.update(userId, { avatar: avatarUrl });

    this.logger.logBusiness('avatar_uploaded', userId, { avatarUrl });

    return avatarUrl;
  }

  async followUser(followData: FollowUserRequest): Promise<void> {
    const { userId, targetUserId } = followData;

    if (userId === targetUserId) {
      throw new ForbiddenException('Cannot follow yourself');
    }

    const targetUser = await this.userRepository.findById(targetUserId);
    if (!targetUser) {
      throw new NotFoundException('User to follow not found');
    }

    // Check if already following
    const existingFollow = await this.followRepository.findByFollowerAndFollowing(userId, targetUserId);
    if (existingFollow) {
      throw new ForbiddenException('Already following this user');
    }

    // Create follow relationship
    await this.followRepository.create({
      followerId: userId,
      followingId: targetUserId,
    });

    // Update follower counts
    await this.userRepository.incrementFollowingCount(userId);
    await this.userRepository.incrementFollowersCount(targetUserId);

    this.logger.logBusiness('user_followed', userId, { targetUserId });
  }

  async unfollowUser(followData: FollowUserRequest): Promise<void> {
    const { userId, targetUserId } = followData;

    const follow = await this.followRepository.findByFollowerAndFollowing(userId, targetUserId);
    if (!follow) {
      throw new ForbiddenException('Not following this user');
    }

    // Delete follow relationship
    await this.followRepository.delete(follow.id);

    // Update follower counts
    await this.userRepository.decrementFollowingCount(userId);
    await this.userRepository.decrementFollowersCount(targetUserId);

    this.logger.logBusiness('user_unfollowed', userId, { targetUserId });
  }

  async getFollowers(userId: string, limit: number = 10, offset: number = 0): Promise<any[]> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    return this.followRepository.findFollowers(userId, limit, offset);
  }

  async getFollowing(userId: string, limit: number = 10, offset: number = 0): Promise<any[]> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    return this.followRepository.findFollowing(userId, limit, offset);
  }

  async searchUsers(query: string, limit: number = 10, offset: number = 0): Promise<User[]> {
    return this.userRepository.searchUsers(query, limit, offset);
  }

  async getTopCreators(limit: number = 10): Promise<User[]> {
    return this.userRepository.findTopCreators(limit);
  }

  async getUsersWithMostFollowers(limit: number = 10): Promise<User[]> {
    return this.userRepository.findUsersWithMostFollowers(limit);
  }

  async isFollowing(userId: string, targetUserId: string): Promise<boolean> {
    return this.followRepository.isFollowing(userId, targetUserId);
  }

  async getFollowStats(userId: string): Promise<{
    followersCount: number;
    followingCount: number;
  }> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    return {
      followersCount: user.followersCount,
      followingCount: user.followingCount,
    };
  }

  async updateUserStats(userId: string, stats: {
    followersCount?: number;
    followingCount?: number;
    postsCount?: number;
    videosCount?: number;
    totalViews?: number;
    totalLikes?: number;
  }): Promise<void> {
    await this.userRepository.updateUserStats(userId, stats);
  }

  async deleteUser(userId: string): Promise<void> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Soft delete user
    await this.userRepository.update(userId, { status: 'banned' });

    this.logger.logBusiness('user_deleted', userId, { email: user.email });
  }

  async banUser(userId: string, reason?: string): Promise<void> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    await this.userRepository.update(userId, { 
      status: 'banned',
      metadata: { ...user.metadata, banReason: reason, bannedAt: new Date() }
    });

    this.logger.logSecurity('user_banned', userId, undefined, { reason });
  }

  async unbanUser(userId: string): Promise<void> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    await this.userRepository.update(userId, { 
      status: 'active',
      metadata: { ...user.metadata, unbannedAt: new Date() }
    });

    this.logger.logSecurity('user_unbanned', userId, undefined, {});
  }
}
