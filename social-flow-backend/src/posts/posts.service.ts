import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { PostRepository } from '../shared/database/repositories/post.repository';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { Post, PostType, PostStatus } from '../shared/database/entities/post.entity';
import { S3Service } from '../shared/aws/s3.service';
import { LoggerService } from '../shared/logger/logger.service';

export interface CreatePostRequest {
  content: string;
  type?: PostType;
  hashtags?: string[];
  mentions?: string[];
  mediaUrls?: string[];
  linkUrl?: string;
  linkTitle?: string;
  linkDescription?: string;
  linkImage?: string;
  pollOptions?: Record<string, any>[];
  pollEndsAt?: Date;
  isSensitive?: boolean;
  isAgeRestricted?: boolean;
  location?: Record<string, any>;
  parentId?: string;
  threadId?: string;
}

export interface UpdatePostRequest {
  content?: string;
  hashtags?: string[];
  mentions?: string[];
  isSensitive?: boolean;
  isAgeRestricted?: boolean;
  location?: Record<string, any>;
}

@Injectable()
export class PostsService {
  constructor(
    private postRepository: PostRepository,
    private userRepository: UserRepository,
    private s3Service: S3Service,
    private logger: LoggerService,
  ) {}

  async createPost(userId: string, postData: CreatePostRequest): Promise<Post> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Extract hashtags and mentions from content
    const hashtags = this.extractHashtags(postData.content);
    const mentions = this.extractMentions(postData.content);

    const post = await this.postRepository.create({
      content: postData.content,
      type: postData.type || PostType.TEXT,
      hashtags: [...hashtags, ...(postData.hashtags || [])],
      mentions: [...mentions, ...(postData.mentions || [])],
      mediaUrls: postData.mediaUrls || [],
      linkUrl: postData.linkUrl,
      linkTitle: postData.linkTitle,
      linkDescription: postData.linkDescription,
      linkImage: postData.linkImage,
      pollOptions: postData.pollOptions || [],
      pollEndsAt: postData.pollEndsAt,
      isSensitive: postData.isSensitive || false,
      isAgeRestricted: postData.isAgeRestricted || false,
      location: postData.location,
      parentId: postData.parentId,
      threadId: postData.threadId,
      userId,
      status: PostStatus.PUBLISHED,
    });

    // Update user post count
    await this.userRepository.incrementPostsCount(userId);

    this.logger.logBusiness('post_created', userId, {
      postId: post.id,
      type: post.type,
      hashtags: post.hashtags,
      mentions: post.mentions,
    });

    return post;
  }

  async getPost(postId: string): Promise<Post> {
    const post = await this.postRepository.findById(postId);
    if (!post) {
      throw new NotFoundException('Post not found');
    }

    return post;
  }

  async updatePost(postId: string, userId: string, updateData: UpdatePostRequest): Promise<Post> {
    const post = await this.postRepository.findById(postId);
    if (!post) {
      throw new NotFoundException('Post not found');
    }

    if (post.userId !== userId) {
      throw new ForbiddenException('Not authorized to update this post');
    }

    // Extract hashtags and mentions from content if content is being updated
    if (updateData.content) {
      const hashtags = this.extractHashtags(updateData.content);
      const mentions = this.extractMentions(updateData.content);
      updateData.hashtags = [...hashtags, ...(updateData.hashtags || [])];
      updateData.mentions = [...mentions, ...(updateData.mentions || [])];
    }

    const updatedPost = await this.postRepository.update(postId, updateData);

    this.logger.logBusiness('post_updated', userId, {
      postId,
      updates: updateData,
    });

    return updatedPost;
  }

  async deletePost(postId: string, userId: string): Promise<void> {
    const post = await this.postRepository.findById(postId);
    if (!post) {
      throw new NotFoundException('Post not found');
    }

    if (post.userId !== userId) {
      throw new ForbiddenException('Not authorized to delete this post');
    }

    // Soft delete post
    await this.postRepository.update(postId, { status: PostStatus.DELETED });

    // Update user post count
    await this.userRepository.decrementPostsCount(userId);

    this.logger.logBusiness('post_deleted', userId, {
      postId,
      content: post.content,
    });
  }

  async getFeed(userId: string, limit: number = 10, offset: number = 0): Promise<Post[]> {
    // Get users that the current user follows
    const following = await this.userRepository.findFollowing(userId, 1000, 0);
    const followingIds = following.map(f => f.followingId);

    // Include current user's posts
    followingIds.push(userId);

    return this.postRepository.findFeedPosts(followingIds, limit, offset);
  }

  async getPostsByHashtag(hashtag: string, limit: number = 10, offset: number = 0): Promise<Post[]> {
    return this.postRepository.findPostsByHashtag(hashtag, limit, offset);
  }

  async searchPosts(query: string, limit: number = 10, offset: number = 0): Promise<Post[]> {
    return this.postRepository.searchPosts(query, limit, offset);
  }

  async getUserPosts(userId: string, limit: number = 10, offset: number = 0): Promise<Post[]> {
    return this.postRepository.findByUserId(userId, limit, offset);
  }

  async getPostStats(postId: string): Promise<{
    likesCount: number;
    commentsCount: number;
    sharesCount: number;
    viewsCount: number;
    retweetsCount: number;
  }> {
    const post = await this.postRepository.findById(postId);
    if (!post) {
      throw new NotFoundException('Post not found');
    }

    return {
      likesCount: post.likesCount,
      commentsCount: post.commentsCount,
      sharesCount: post.sharesCount,
      viewsCount: post.viewsCount,
      retweetsCount: post.retweetsCount,
    };
  }

  async updatePostStats(postId: string, stats: {
    likesCount?: number;
    commentsCount?: number;
    sharesCount?: number;
    viewsCount?: number;
    retweetsCount?: number;
  }): Promise<void> {
    await this.postRepository.update(postId, stats);
  }

  async incrementLikesCount(postId: string): Promise<void> {
    await this.postRepository.incrementLikesCount(postId);
  }

  async decrementLikesCount(postId: string): Promise<void> {
    await this.postRepository.decrementLikesCount(postId);
  }

  async incrementCommentsCount(postId: string): Promise<void> {
    await this.postRepository.incrementCommentsCount(postId);
  }

  async decrementCommentsCount(postId: string): Promise<void> {
    await this.postRepository.decrementCommentsCount(postId);
  }

  async incrementSharesCount(postId: string): Promise<void> {
    await this.postRepository.incrementSharesCount(postId);
  }

  async incrementViewsCount(postId: string): Promise<void> {
    await this.postRepository.incrementViewsCount(postId);
  }

  private extractHashtags(content: string): string[] {
    const hashtagRegex = /#[\w\u0590-\u05ff]+/g;
    const matches = content.match(hashtagRegex);
    return matches ? matches.map(tag => tag.substring(1).toLowerCase()) : [];
  }

  private extractMentions(content: string): string[] {
    const mentionRegex = /@[\w\u0590-\u05ff]+/g;
    const matches = content.match(mentionRegex);
    return matches ? matches.map(mention => mention.substring(1).toLowerCase()) : [];
  }
}
