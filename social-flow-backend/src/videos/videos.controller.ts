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
  UploadedFile,
  UseInterceptors,
  ValidationPipe,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiConsumes } from '@nestjs/swagger';
import { ThrottlerGuard } from '@nestjs/throttler';

import { VideosService, UploadVideoRequest, UpdateVideoRequest } from './videos.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('videos')
@Controller('videos')
@UseGuards(ThrottlerGuard)
export class VideosController {
  constructor(private readonly videosService: VideosService) {}

  @Post('upload')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiConsumes('multipart/form-data')
  @ApiOperation({ summary: 'Upload a video' })
  @ApiResponse({ status: 201, description: 'Video uploaded successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  @UseInterceptors(FileInterceptor('video'))
  async uploadVideo(
    @Body(ValidationPipe) uploadData: UploadVideoRequest,
    @UploadedFile() file: Express.Multer.File,
    @Request() req,
  ) {
    return this.videosService.uploadVideo(req.user.id, uploadData, file);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get video by ID' })
  @ApiResponse({ status: 200, description: 'Video retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Video not found' })
  @ApiResponse({ status: 403, description: 'Video is private' })
  async getVideo(@Param('id') videoId: string, @Request() req) {
    return this.videosService.getVideo(videoId, req.user?.id);
  }

  @Put(':id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update video' })
  @ApiResponse({ status: 200, description: 'Video updated successfully' })
  @ApiResponse({ status: 404, description: 'Video not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to update this video' })
  async updateVideo(
    @Param('id') videoId: string,
    @Body(ValidationPipe) updateData: UpdateVideoRequest,
    @Request() req,
  ) {
    return this.videosService.updateVideo(videoId, req.user.id, updateData);
  }

  @Delete(':id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Delete video' })
  @ApiResponse({ status: 200, description: 'Video deleted successfully' })
  @ApiResponse({ status: 404, description: 'Video not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to delete this video' })
  async deleteVideo(@Param('id') videoId: string, @Request() req) {
    return this.videosService.deleteVideo(videoId, req.user.id);
  }

  @Get(':id/stream')
  @ApiOperation({ summary: 'Get video stream URL' })
  @ApiResponse({ status: 200, description: 'Stream URL retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Video not found' })
  @ApiResponse({ status: 403, description: 'Video is private' })
  @ApiResponse({ status: 400, description: 'Video is still processing' })
  async getVideoStream(@Param('id') videoId: string, @Request() req) {
    const streamUrl = await this.videosService.getVideoStream(videoId, req.user?.id);
    return { streamUrl };
  }

  @Get(':id/thumbnail')
  @ApiOperation({ summary: 'Get video thumbnail' })
  @ApiResponse({ status: 200, description: 'Thumbnail URL retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Video not found' })
  async getVideoThumbnail(
    @Param('id') videoId: string,
    @Query('size') size: string = 'default',
  ) {
    const thumbnailUrl = await this.videosService.getVideoThumbnail(videoId, size);
    return { thumbnailUrl };
  }

  @Post(':id/view')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Record video view' })
  @ApiResponse({ status: 200, description: 'View recorded successfully' })
  @ApiResponse({ status: 404, description: 'Video not found' })
  async recordView(@Param('id') videoId: string, @Request() req) {
    return this.videosService.recordView(videoId, req.user?.id);
  }

  @Get('trending')
  @ApiOperation({ summary: 'Get trending videos' })
  @ApiResponse({ status: 200, description: 'Trending videos retrieved successfully' })
  async getTrendingVideos(
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.videosService.getTrendingVideos(limit, offset);
  }

  @Get('tag/:tag')
  @ApiOperation({ summary: 'Get videos by tag' })
  @ApiResponse({ status: 200, description: 'Videos retrieved successfully' })
  async getVideosByTag(
    @Param('tag') tag: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.videosService.getVideosByTag(tag, limit, offset);
  }

  @Get('search')
  @ApiOperation({ summary: 'Search videos' })
  @ApiResponse({ status: 200, description: 'Videos retrieved successfully' })
  async searchVideos(
    @Query('q') query: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.videosService.searchVideos(query, limit, offset);
  }

  @Get('user/:userId')
  @ApiOperation({ summary: 'Get user videos' })
  @ApiResponse({ status: 200, description: 'Videos retrieved successfully' })
  async getUserVideos(
    @Param('userId') userId: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.videosService.getUserVideos(userId, limit, offset);
  }

  @Get('public')
  @ApiOperation({ summary: 'Get public videos' })
  @ApiResponse({ status: 200, description: 'Videos retrieved successfully' })
  async getPublicVideos(
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.videosService.getPublicVideos(limit, offset);
  }

  @Get(':id/stats')
  @ApiOperation({ summary: 'Get video statistics' })
  @ApiResponse({ status: 200, description: 'Stats retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Video not found' })
  async getVideoStats(@Param('id') videoId: string) {
    return this.videosService.getVideoStats(videoId);
  }
}
