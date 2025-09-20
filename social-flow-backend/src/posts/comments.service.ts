import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { CommentRepository } from '../shared/database/repositories/comment.repository';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { Comment } from '../shared/database/entities/comment.entity';
import { LoggerService } from '../shared/logger/logger.service';

export interface CreateCommentRequest {
  content: string;
  postId?: string;
  videoId?: string;
  parentId?: string;
}

export interface UpdateCommentRequest {
  content: string;
}

@Injectable()
export class CommentsService {
  constructor(
    private commentRepository: CommentRepository,
    private userRepository: UserRepository,
    private logger: LoggerService,
  ) {}

  async createComment(userId: string, commentData: CreateCommentRequest): Promise<Comment> {
    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    if (!commentData.postId && !commentData.videoId) {
      throw new ForbiddenException('Comment must be associated with a post or video');
    }

    const comment = await this.commentRepository.create({
      content: commentData.content,
      postId: commentData.postId,
      videoId: commentData.videoId,
      parentId: commentData.parentId,
      userId,
    });

    // Update comment count on post or video
    if (commentData.postId) {
      await this.commentRepository.incrementCommentsCount(commentData.postId);
    }
    if (commentData.videoId) {
      await this.commentRepository.incrementCommentsCount(commentData.videoId);
    }

    // Update parent comment replies count if it's a reply
    if (commentData.parentId) {
      await this.commentRepository.incrementRepliesCount(commentData.parentId);
    }

    this.logger.logBusiness('comment_created', userId, {
      commentId: comment.id,
      postId: commentData.postId,
      videoId: commentData.videoId,
      parentId: commentData.parentId,
    });

    return comment;
  }

  async getComment(commentId: string): Promise<Comment> {
    const comment = await this.commentRepository.findById(commentId);
    if (!comment) {
      throw new NotFoundException('Comment not found');
    }

    return comment;
  }

  async updateComment(commentId: string, userId: string, updateData: UpdateCommentRequest): Promise<Comment> {
    const comment = await this.commentRepository.findById(commentId);
    if (!comment) {
      throw new NotFoundException('Comment not found');
    }

    if (comment.userId !== userId) {
      throw new ForbiddenException('Not authorized to update this comment');
    }

    const updatedComment = await this.commentRepository.update(commentId, {
      content: updateData.content,
      isEdited: true,
      editedAt: new Date(),
    });

    this.logger.logBusiness('comment_updated', userId, {
      commentId,
      updates: updateData,
    });

    return updatedComment;
  }

  async deleteComment(commentId: string, userId: string): Promise<void> {
    const comment = await this.commentRepository.findById(commentId);
    if (!comment) {
      throw new NotFoundException('Comment not found');
    }

    if (comment.userId !== userId) {
      throw new ForbiddenException('Not authorized to delete this comment');
    }

    // Update comment count on post or video
    if (comment.postId) {
      await this.commentRepository.decrementCommentsCount(comment.postId);
    }
    if (comment.videoId) {
      await this.commentRepository.decrementCommentsCount(comment.videoId);
    }

    // Update parent comment replies count if it's a reply
    if (comment.parentId) {
      await this.commentRepository.decrementRepliesCount(comment.parentId);
    }

    // Delete comment
    await this.commentRepository.delete(commentId);

    this.logger.logBusiness('comment_deleted', userId, {
      commentId,
      content: comment.content,
    });
  }

  async getPostComments(postId: string, limit: number = 10, offset: number = 0): Promise<Comment[]> {
    return this.commentRepository.findByPostId(postId, limit, offset);
  }

  async getVideoComments(videoId: string, limit: number = 10, offset: number = 0): Promise<Comment[]> {
    return this.commentRepository.findByVideoId(videoId, limit, offset);
  }

  async getCommentReplies(commentId: string, limit: number = 10, offset: number = 0): Promise<Comment[]> {
    return this.commentRepository.findReplies(commentId, limit, offset);
  }

  async getCommentStats(commentId: string): Promise<{
    likesCount: number;
    repliesCount: number;
  }> {
    const comment = await this.commentRepository.findById(commentId);
    if (!comment) {
      throw new NotFoundException('Comment not found');
    }

    return {
      likesCount: comment.likesCount,
      repliesCount: comment.repliesCount,
    };
  }

  async updateCommentStats(commentId: string, stats: {
    likesCount?: number;
    repliesCount?: number;
  }): Promise<void> {
    await this.commentRepository.update(commentId, stats);
  }

  async incrementLikesCount(commentId: string): Promise<void> {
    await this.commentRepository.incrementLikesCount(commentId);
  }

  async decrementLikesCount(commentId: string): Promise<void> {
    await this.commentRepository.decrementLikesCount(commentId);
  }

  async incrementRepliesCount(commentId: string): Promise<void> {
    await this.commentRepository.incrementRepliesCount(commentId);
  }

  async decrementRepliesCount(commentId: string): Promise<void> {
    await this.commentRepository.decrementRepliesCount(commentId);
  }
}
