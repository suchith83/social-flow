import { Injectable, NotFoundException, ForbiddenException, ConflictException } from '@nestjs/common';
import { LikeRepository } from '../shared/database/repositories/like.repository';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { Like, LikeType } from '../shared/database/entities/like.entity';
import { LoggerService } from '../shared/logger/logger.service';

export interface CreateLikeRequest {
  postId?: string;
  videoId?: string;
  commentId?: string;
  type?: LikeType;
}

@Injectable()
export class LikesService {
  constructor(
    private likeRepository: LikeRepository,
    private userRepository: UserRepository,
    private logger: LoggerService,
  ) {}

  async createLike(userId: string, likeData: CreateLikeRequest): Promise<Like> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    if (!likeData.postId && !likeData.videoId && !likeData.commentId) {
      throw new ForbiddenException('Like must be associated with a post, video, or comment');
    }

    // Check if like already exists
    let existingLike: Like | null = null;
    if (likeData.postId) {
      existingLike = await this.likeRepository.findByUserAndPost(userId, likeData.postId);
    } else if (likeData.videoId) {
      existingLike = await this.likeRepository.findByUserAndVideo(userId, likeData.videoId);
    } else if (likeData.commentId) {
      existingLike = await this.likeRepository.findByUserAndComment(userId, likeData.commentId);
    }

    if (existingLike) {
      throw new ConflictException('Already liked this item');
    }

    const like = await this.likeRepository.create({
      userId,
      postId: likeData.postId,
      videoId: likeData.videoId,
      commentId: likeData.commentId,
      type: likeData.type || LikeType.LIKE,
    });

    // Update like count on post, video, or comment
    if (likeData.postId) {
      await this.likeRepository.incrementLikesCount(likeData.postId);
    }
    if (likeData.videoId) {
      await this.likeRepository.incrementLikesCount(likeData.videoId);
    }
    if (likeData.commentId) {
      await this.likeRepository.incrementLikesCount(likeData.commentId);
    }

    this.logger.logBusiness('like_created', userId, {
      likeId: like.id,
      postId: likeData.postId,
      videoId: likeData.videoId,
      commentId: likeData.commentId,
      type: like.type,
    });

    return like;
  }

  async removeLike(userId: string, likeData: CreateLikeRequest): Promise<void> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Find existing like
    let existingLike: Like | null = null;
    if (likeData.postId) {
      existingLike = await this.likeRepository.findByUserAndPost(userId, likeData.postId);
    } else if (likeData.videoId) {
      existingLike = await this.likeRepository.findByUserAndVideo(userId, likeData.videoId);
    } else if (likeData.commentId) {
      existingLike = await this.likeRepository.findByUserAndComment(userId, likeData.commentId);
    }

    if (!existingLike) {
      throw new NotFoundException('Like not found');
    }

    // Update like count on post, video, or comment
    if (likeData.postId) {
      await this.likeRepository.decrementLikesCount(likeData.postId);
    }
    if (likeData.videoId) {
      await this.likeRepository.decrementLikesCount(likeData.videoId);
    }
    if (likeData.commentId) {
      await this.likeRepository.decrementLikesCount(likeData.commentId);
    }

    // Delete like
    await this.likeRepository.delete(existingLike.id);

    this.logger.logBusiness('like_removed', userId, {
      likeId: existingLike.id,
      postId: likeData.postId,
      videoId: likeData.videoId,
      commentId: likeData.commentId,
    });
  }

  async getLikes(postId?: string, videoId?: string, commentId?: string, limit: number = 10, offset: number = 0): Promise<Like[]> {
    if (postId) {
      return this.likeRepository.findLikesByPost(postId, limit, offset);
    }
    if (videoId) {
      return this.likeRepository.findLikesByVideo(videoId, limit, offset);
    }
    if (commentId) {
      return this.likeRepository.findLikesByComment(commentId, limit, offset);
    }

    throw new ForbiddenException('Must specify postId, videoId, or commentId');
  }

  async getLikeCount(postId?: string, videoId?: string, commentId?: string): Promise<number> {
    if (postId) {
      return this.likeRepository.countLikesByPost(postId);
    }
    if (videoId) {
      return this.likeRepository.countLikesByVideo(videoId);
    }
    if (commentId) {
      return this.likeRepository.countLikesByComment(commentId);
    }

    throw new ForbiddenException('Must specify postId, videoId, or commentId');
  }

  async isLiked(userId: string, postId?: string, videoId?: string, commentId?: string): Promise<boolean> {
    if (postId) {
      const like = await this.likeRepository.findByUserAndPost(userId, postId);
      return !!like;
    }
    if (videoId) {
      const like = await this.likeRepository.findByUserAndVideo(userId, videoId);
      return !!like;
    }
    if (commentId) {
      const like = await this.likeRepository.findByUserAndComment(userId, commentId);
      return !!like;
    }

    throw new ForbiddenException('Must specify postId, videoId, or commentId');
  }

  async getUserLikes(userId: string, limit: number = 10, offset: number = 0): Promise<Like[]> {
    return this.likeRepository.findByUser(userId, limit, offset);
  }

  async getLikeStats(userId: string): Promise<{
    totalLikes: number;
    likesByType: Record<string, number>;
  }> {
    const totalLikes = await this.likeRepository.countLikesByUser(userId);
    
    // Get likes by type (this would require a more complex query in production)
    const likesByType = {
      [LikeType.LIKE]: 0,
      [LikeType.LOVE]: 0,
      [LikeType.LAUGH]: 0,
      [LikeType.ANGRY]: 0,
      [LikeType.SAD]: 0,
      [LikeType.WOW]: 0,
    };

    return {
      totalLikes,
      likesByType,
    };
  }
}
