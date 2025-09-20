import { Injectable, NotFoundException } from '@nestjs/common';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { VideoRepository } from '../shared/database/repositories/video.repository';
import { PostRepository } from '../shared/database/repositories/post.repository';
import { LikeRepository } from '../shared/database/repositories/like.repository';
import { FollowRepository } from '../shared/database/repositories/follow.repository';
import { ViewCountRepository } from '../shared/database/repositories/view-count.repository';
import { LoggerService } from '../shared/logger/logger.service';

export interface RecommendationRequest {
  userId: string;
  type: 'videos' | 'posts' | 'users';
  limit?: number;
  filters?: Record<string, any>;
}

export interface RecommendationResult {
  recommendations: any[];
  total: number;
  algorithm: string;
  confidence: number;
}

@Injectable()
export class RecommendationService {
  constructor(
    private userRepository: UserRepository,
    private videoRepository: VideoRepository,
    private postRepository: PostRepository,
    private likeRepository: LikeRepository,
    private followRepository: FollowRepository,
    private viewCountRepository: ViewCountRepository,
    private logger: LoggerService,
  ) {}

  async getRecommendations(request: RecommendationRequest): Promise<RecommendationResult> {
    try {
      const { userId, type, limit = 20, filters = {} } = request;
      
      let recommendations: any[] = [];
      let algorithm = 'collaborative';
      let confidence = 0.8;
      
      switch (type) {
        case 'videos':
          recommendations = await this.getVideoRecommendations(userId, limit, filters);
          break;
        case 'posts':
          recommendations = await this.getPostRecommendations(userId, limit, filters);
          break;
        case 'users':
          recommendations = await this.getUserRecommendations(userId, limit, filters);
          break;
        default:
          throw new NotFoundException('Invalid recommendation type');
      }
      
      return {
        recommendations,
        total: recommendations.length,
        algorithm,
        confidence,
      };
    } catch (error) {
      this.logger.logError(error, 'RecommendationService.getRecommendations', { request });
      throw error;
    }
  }

  async getVideoRecommendations(userId: string, limit: number, filters: Record<string, any>): Promise<any[]> {
    try {
      // Get user's liked videos
      const likedVideos = await this.likeRepository.findByUser(userId, 100, 0);
      const likedVideoIds = likedVideos.map(like => like.videoId).filter(id => id);
      
      // Get user's followed users
      const followedUsers = await this.followRepository.findByFollower(userId, 100, 0);
      const followedUserIds = followedUsers.map(follow => follow.followingId);
      
      // Get user's viewed videos
      const viewedVideos = await this.viewCountRepository.findByUser(userId, 100, 0);
      const viewedVideoIds = viewedVideos.map(view => view.videoId);
      
      // Get recommendations based on collaborative filtering
      const recommendations = await this.getCollaborativeVideoRecommendations(
        userId,
        likedVideoIds,
        followedUserIds,
        viewedVideoIds,
        limit,
        filters,
      );
      
      return recommendations;
    } catch (error) {
      this.logger.logError(error, 'RecommendationService.getVideoRecommendations', { userId, limit, filters });
      throw error;
    }
  }

  async getPostRecommendations(userId: string, limit: number, filters: Record<string, any>): Promise<any[]> {
    try {
      // Get user's liked posts
      const likedPosts = await this.likeRepository.findByUser(userId, 100, 0);
      const likedPostIds = likedPosts.map(like => like.postId).filter(id => id);
      
      // Get user's followed users
      const followedUsers = await this.followRepository.findByFollower(userId, 100, 0);
      const followedUserIds = followedUsers.map(follow => follow.followingId);
      
      // Get recommendations based on collaborative filtering
      const recommendations = await this.getCollaborativePostRecommendations(
        userId,
        likedPostIds,
        followedUserIds,
        limit,
        filters,
      );
      
      return recommendations;
    } catch (error) {
      this.logger.logError(error, 'RecommendationService.getPostRecommendations', { userId, limit, filters });
      throw error;
    }
  }

  async getUserRecommendations(userId: string, limit: number, filters: Record<string, any>): Promise<any[]> {
    try {
      // Get user's followed users
      const followedUsers = await this.followRepository.findByFollower(userId, 100, 0);
      const followedUserIds = followedUsers.map(follow => follow.followingId);
      
      // Get recommendations based on social connections
      const recommendations = await this.getSocialUserRecommendations(
        userId,
        followedUserIds,
        limit,
        filters,
      );
      
      return recommendations;
    } catch (error) {
      this.logger.logError(error, 'RecommendationService.getUserRecommendations', { userId, limit, filters });
      throw error;
    }
  }

  async getTrendingVideos(limit: number = 20): Promise<any[]> {
    try {
      const videos = await this.videoRepository.findTrending(limit);
      return videos;
    } catch (error) {
      this.logger.logError(error, 'RecommendationService.getTrendingVideos', { limit });
      throw error;
    }
  }

  async getTrendingPosts(limit: number = 20): Promise<any[]> {
    try {
      const posts = await this.postRepository.findTrending(limit);
      return posts;
    } catch (error) {
      this.logger.logError(error, 'RecommendationService.getTrendingPosts', { limit });
      throw error;
    }
  }

  async getTrendingUsers(limit: number = 20): Promise<any[]> {
    try {
      const users = await this.userRepository.findTrending(limit);
      return users;
    } catch (error) {
      this.logger.logError(error, 'RecommendationService.getTrendingUsers', { limit });
      throw error;
    }
  }

  private async getCollaborativeVideoRecommendations(
    userId: string,
    likedVideoIds: string[],
    followedUserIds: string[],
    viewedVideoIds: string[],
    limit: number,
    filters: Record<string, any>,
  ): Promise<any[]> {
    // This would implement collaborative filtering algorithm
    // For now, return trending videos
    return this.getTrendingVideos(limit);
  }

  private async getCollaborativePostRecommendations(
    userId: string,
    likedPostIds: string[],
    followedUserIds: string[],
    limit: number,
    filters: Record<string, any>,
  ): Promise<any[]> {
    // This would implement collaborative filtering algorithm
    // For now, return trending posts
    return this.getTrendingPosts(limit);
  }

  private async getSocialUserRecommendations(
    userId: string,
    followedUserIds: string[],
    limit: number,
    filters: Record<string, any>,
  ): Promise<any[]> {
    // This would implement social recommendation algorithm
    // For now, return trending users
    return this.getTrendingUsers(limit);
  }
}
