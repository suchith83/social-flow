import {
  Controller,
  Get,
  Post,
  Put,
  Delete,
  Body,
  Param,
  Query,
  UseGuards,
  Request,
  ValidationPipe,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { ThrottlerGuard } from '@nestjs/throttler';

import { PostsService, CreatePostRequest, UpdatePostRequest } from './posts.service';
import { CommentsService, CreateCommentRequest, UpdateCommentRequest } from './comments.service';
import { LikesService, CreateLikeRequest } from './likes.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('posts')
@Controller('posts')
@UseGuards(ThrottlerGuard)
export class PostsController {
  constructor(
    private readonly postsService: PostsService,
    private readonly commentsService: CommentsService,
    private readonly likesService: LikesService,
  ) {}

  // Posts endpoints
  @Post()
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Create a new post' })
  @ApiResponse({ status: 201, description: 'Post created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  async createPost(
    @Body(ValidationPipe) postData: CreatePostRequest,
    @Request() req,
  ) {
    return this.postsService.createPost(req.user.id, postData);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get post by ID' })
  @ApiResponse({ status: 200, description: 'Post retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Post not found' })
  async getPost(@Param('id') postId: string) {
    return this.postsService.getPost(postId);
  }

  @Put(':id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update post' })
  @ApiResponse({ status: 200, description: 'Post updated successfully' })
  @ApiResponse({ status: 404, description: 'Post not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to update this post' })
  async updatePost(
    @Param('id') postId: string,
    @Body(ValidationPipe) updateData: UpdatePostRequest,
    @Request() req,
  ) {
    return this.postsService.updatePost(postId, req.user.id, updateData);
  }

  @Delete(':id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Delete post' })
  @ApiResponse({ status: 200, description: 'Post deleted successfully' })
  @ApiResponse({ status: 404, description: 'Post not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to delete this post' })
  async deletePost(@Param('id') postId: string, @Request() req) {
    return this.postsService.deletePost(postId, req.user.id);
  }

  @Get('feed')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get user feed' })
  @ApiResponse({ status: 200, description: 'Feed retrieved successfully' })
  async getFeed(
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
    @Request() req,
  ) {
    return this.postsService.getFeed(req.user.id, limit, offset);
  }

  @Get('hashtag/:hashtag')
  @ApiOperation({ summary: 'Get posts by hashtag' })
  @ApiResponse({ status: 200, description: 'Posts retrieved successfully' })
  async getPostsByHashtag(
    @Param('hashtag') hashtag: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.postsService.getPostsByHashtag(hashtag, limit, offset);
  }

  @Get('search')
  @ApiOperation({ summary: 'Search posts' })
  @ApiResponse({ status: 200, description: 'Posts retrieved successfully' })
  async searchPosts(
    @Query('q') query: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.postsService.searchPosts(query, limit, offset);
  }

  @Get('user/:userId')
  @ApiOperation({ summary: 'Get user posts' })
  @ApiResponse({ status: 200, description: 'Posts retrieved successfully' })
  async getUserPosts(
    @Param('userId') userId: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.postsService.getUserPosts(userId, limit, offset);
  }

  @Get(':id/stats')
  @ApiOperation({ summary: 'Get post statistics' })
  @ApiResponse({ status: 200, description: 'Stats retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Post not found' })
  async getPostStats(@Param('id') postId: string) {
    return this.postsService.getPostStats(postId);
  }

  // Comments endpoints
  @Post(':id/comments')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Create a comment on a post' })
  @ApiResponse({ status: 201, description: 'Comment created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  async createComment(
    @Param('id') postId: string,
    @Body(ValidationPipe) commentData: CreateCommentRequest,
    @Request() req,
  ) {
    return this.commentsService.createComment(req.user.id, {
      ...commentData,
      postId,
    });
  }

  @Get(':id/comments')
  @ApiOperation({ summary: 'Get post comments' })
  @ApiResponse({ status: 200, description: 'Comments retrieved successfully' })
  async getPostComments(
    @Param('id') postId: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.commentsService.getPostComments(postId, limit, offset);
  }

  @Put('comments/:commentId')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update comment' })
  @ApiResponse({ status: 200, description: 'Comment updated successfully' })
  @ApiResponse({ status: 404, description: 'Comment not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to update this comment' })
  async updateComment(
    @Param('commentId') commentId: string,
    @Body(ValidationPipe) updateData: UpdateCommentRequest,
    @Request() req,
  ) {
    return this.commentsService.updateComment(commentId, req.user.id, updateData);
  }

  @Delete('comments/:commentId')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Delete comment' })
  @ApiResponse({ status: 200, description: 'Comment deleted successfully' })
  @ApiResponse({ status: 404, description: 'Comment not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to delete this comment' })
  async deleteComment(@Param('commentId') commentId: string, @Request() req) {
    return this.commentsService.deleteComment(commentId, req.user.id);
  }

  @Get('comments/:commentId/replies')
  @ApiOperation({ summary: 'Get comment replies' })
  @ApiResponse({ status: 200, description: 'Replies retrieved successfully' })
  async getCommentReplies(
    @Param('commentId') commentId: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.commentsService.getCommentReplies(commentId, limit, offset);
  }

  // Likes endpoints
  @Post(':id/like')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Like a post' })
  @ApiResponse({ status: 201, description: 'Post liked successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  @ApiResponse({ status: 409, description: 'Already liked this post' })
  async likePost(
    @Param('id') postId: string,
    @Body(ValidationPipe) likeData: CreateLikeRequest,
    @Request() req,
  ) {
    return this.likesService.createLike(req.user.id, {
      ...likeData,
      postId,
    });
  }

  @Delete(':id/like')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Unlike a post' })
  @ApiResponse({ status: 200, description: 'Post unliked successfully' })
  @ApiResponse({ status: 404, description: 'Like not found' })
  async unlikePost(
    @Param('id') postId: string,
    @Request() req,
  ) {
    return this.likesService.removeLike(req.user.id, { postId });
  }

  @Get(':id/likes')
  @ApiOperation({ summary: 'Get post likes' })
  @ApiResponse({ status: 200, description: 'Likes retrieved successfully' })
  async getPostLikes(
    @Param('id') postId: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.likesService.getLikes(postId, undefined, undefined, limit, offset);
  }

  @Get(':id/like-status')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Check if user liked a post' })
  @ApiResponse({ status: 200, description: 'Like status retrieved successfully' })
  async getLikeStatus(@Param('id') postId: string, @Request() req) {
    const isLiked = await this.likesService.isLiked(req.user.id, postId);
    return { isLiked };
  }

  @Get(':id/like-count')
  @ApiOperation({ summary: 'Get post like count' })
  @ApiResponse({ status: 200, description: 'Like count retrieved successfully' })
  async getLikeCount(@Param('id') postId: string) {
    const count = await this.likesService.getLikeCount(postId);
    return { count };
  }
}
