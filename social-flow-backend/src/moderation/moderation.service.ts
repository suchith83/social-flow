import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { VideoRepository } from '../shared/database/repositories/video.repository';
import { PostRepository } from '../shared/database/repositories/post.repository';
import { CommentRepository } from '../shared/database/repositories/comment.repository';
import { LoggerService } from '../shared/logger/logger.service';

export interface ContentModerationRequest {
  entityType: 'video' | 'post' | 'comment' | 'user';
  entityId: string;
  action: 'approve' | 'reject' | 'flag' | 'unflag' | 'ban' | 'unban';
  reason?: string;
  moderatorId: string;
}

export interface ContentModerationResult {
  success: boolean;
  message: string;
  entity: any;
}

@Injectable()
export class ModerationService {
  constructor(
    private userRepository: UserRepository,
    private videoRepository: VideoRepository,
    private postRepository: PostRepository,
    private commentRepository: CommentRepository,
    private logger: LoggerService,
  ) {}

  async moderateContent(request: ContentModerationRequest): Promise<ContentModerationResult> {
    try {
      const { entityType, entityId, action, reason, moderatorId } = request;
      
      let entity;
      let repository;
      
      switch (entityType) {
        case 'video':
          entity = await this.videoRepository.findById(entityId);
          repository = this.videoRepository;
          break;
        case 'post':
          entity = await this.postRepository.findById(entityId);
          repository = this.postRepository;
          break;
        case 'comment':
          entity = await this.commentRepository.findById(entityId);
          repository = this.commentRepository;
          break;
        case 'user':
          entity = await this.userRepository.findById(entityId);
          repository = this.userRepository;
          break;
        default:
          throw new BadRequestException('Invalid entity type');
      }
      
      if (!entity) {
        throw new NotFoundException('Entity not found');
      }
      
      switch (action) {
        case 'approve':
          await this.approveContent(entity, repository, moderatorId);
          break;
        case 'reject':
          await this.rejectContent(entity, repository, reason, moderatorId);
          break;
        case 'flag':
          await this.flagContent(entity, repository, reason, moderatorId);
          break;
        case 'unflag':
          await this.unflagContent(entity, repository, moderatorId);
          break;
        case 'ban':
          await this.banContent(entity, repository, reason, moderatorId);
          break;
        case 'unban':
          await this.unbanContent(entity, repository, moderatorId);
          break;
        default:
          throw new BadRequestException('Invalid action');
      }
      
      this.logger.logBusiness('content_moderated', entityId, {
        entityType,
        action,
        reason,
        moderatorId,
      });
      
      return {
        success: true,
        message: 'Content moderated successfully',
        entity,
      };
    } catch (error) {
      this.logger.logError(error, 'ModerationService.moderateContent', { request });
      throw error;
    }
  }

  async getFlaggedContent(page: number = 1, limit: number = 20): Promise<{ content: any[]; total: number }> {
    try {
      const offset = (page - 1) * limit;
      
      const [flaggedVideos, flaggedPosts, flaggedComments] = await Promise.all([
        this.videoRepository.findFlagged(limit, offset),
        this.postRepository.findFlagged(limit, offset),
        this.commentRepository.findFlagged(limit, offset),
      ]);
      
      const content = [
        ...flaggedVideos.map(video => ({ ...video, type: 'video' })),
        ...flaggedPosts.map(post => ({ ...post, type: 'post' })),
        ...flaggedComments.map(comment => ({ ...comment, type: 'comment' })),
      ];
      
      const total = flaggedVideos.length + flaggedPosts.length + flaggedComments.length;
      
      return { content, total };
    } catch (error) {
      this.logger.logError(error, 'ModerationService.getFlaggedContent', { page, limit });
      throw error;
    }
  }

  async getModerationQueue(page: number = 1, limit: number = 20): Promise<{ content: any[]; total: number }> {
    try {
      const offset = (page - 1) * limit;
      
      const [pendingVideos, pendingPosts, pendingComments] = await Promise.all([
        this.videoRepository.findPending(limit, offset),
        this.postRepository.findPending(limit, offset),
        this.commentRepository.findPending(limit, offset),
      ]);
      
      const content = [
        ...pendingVideos.map(video => ({ ...video, type: 'video' })),
        ...pendingPosts.map(post => ({ ...post, type: 'post' })),
        ...pendingComments.map(comment => ({ ...comment, type: 'comment' })),
      ];
      
      const total = pendingVideos.length + pendingPosts.length + pendingComments.length;
      
      return { content, total };
    } catch (error) {
      this.logger.logError(error, 'ModerationService.getModerationQueue', { page, limit });
      throw error;
    }
  }

  async getModerationStats(): Promise<{
    totalFlagged: number;
    totalPending: number;
    totalApproved: number;
    totalRejected: number;
    totalBanned: number;
  }> {
    try {
      const [
        totalFlagged,
        totalPending,
        totalApproved,
        totalRejected,
        totalBanned,
      ] = await Promise.all([
        this.getTotalFlagged(),
        this.getTotalPending(),
        this.getTotalApproved(),
        this.getTotalRejected(),
        this.getTotalBanned(),
      ]);
      
      return {
        totalFlagged,
        totalPending,
        totalApproved,
        totalRejected,
        totalBanned,
      };
    } catch (error) {
      this.logger.logError(error, 'ModerationService.getModerationStats');
      throw error;
    }
  }

  private async approveContent(entity: any, repository: any, moderatorId: string): Promise<void> {
    entity.isApproved = true;
    entity.approvedAt = new Date();
    entity.approvedBy = moderatorId;
    entity.isRejected = false;
    entity.rejectedAt = null;
    entity.rejectionReason = null;
    entity.rejectedBy = null;
    await repository.update(entity.id, entity);
  }

  private async rejectContent(entity: any, repository: any, reason?: string, moderatorId?: string): Promise<void> {
    entity.isRejected = true;
    entity.rejectedAt = new Date();
    entity.rejectionReason = reason;
    entity.rejectedBy = moderatorId;
    entity.isApproved = false;
    entity.approvedAt = null;
    entity.approvedBy = null;
    await repository.update(entity.id, entity);
  }

  private async flagContent(entity: any, repository: any, reason?: string, moderatorId?: string): Promise<void> {
    entity.isFlagged = true;
    entity.flaggedAt = new Date();
    entity.flagReason = reason;
    entity.flaggedBy = moderatorId;
    await repository.update(entity.id, entity);
  }

  private async unflagContent(entity: any, repository: any, moderatorId?: string): Promise<void> {
    entity.isFlagged = false;
    entity.flaggedAt = null;
    entity.flagReason = null;
    entity.flaggedBy = null;
    await repository.update(entity.id, entity);
  }

  private async banContent(entity: any, repository: any, reason?: string, moderatorId?: string): Promise<void> {
    entity.isBanned = true;
    entity.bannedAt = new Date();
    entity.banReason = reason;
    entity.bannedBy = moderatorId;
    await repository.update(entity.id, entity);
  }

  private async unbanContent(entity: any, repository: any, moderatorId?: string): Promise<void> {
    entity.isBanned = false;
    entity.bannedAt = null;
    entity.banReason = null;
    entity.bannedBy = null;
    await repository.update(entity.id, entity);
  }

  private async getTotalFlagged(): Promise<number> {
    // This would need to be implemented with actual counts
    return 0;
  }

  private async getTotalPending(): Promise<number> {
    // This would need to be implemented with actual counts
    return 0;
  }

  private async getTotalApproved(): Promise<number> {
    // This would need to be implemented with actual counts
    return 0;
  }

  private async getTotalRejected(): Promise<number> {
    // This would need to be implemented with actual counts
    return 0;
  }

  private async getTotalBanned(): Promise<number> {
    // This would need to be implemented with actual counts
    return 0;
  }
}
