import { Controller, Get, Post, Body, Query, Param, UseGuards, Request } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdminService, UserManagementRequest, ContentModerationRequest, AdManagementRequest } from './admin.service';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiQuery } from '@nestjs/swagger';

@ApiTags('admin')
@Controller('admin')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class AdminController {
  constructor(private adminService: AdminService) {}

  @Get('stats')
  @ApiOperation({ summary: 'Get admin statistics' })
  @ApiResponse({ status: 200, description: 'Admin statistics retrieved successfully' })
  async getAdminStats() {
    return this.adminService.getAdminStats();
  }

  @Post('users/manage')
  @ApiOperation({ summary: 'Manage user (ban, unban, suspend, unsuspend, delete)' })
  @ApiResponse({ status: 200, description: 'User managed successfully' })
  async manageUser(@Body() request: UserManagementRequest) {
    await this.adminService.manageUser(request);
    return { message: 'User managed successfully' };
  }

  @Post('content/moderate')
  @ApiOperation({ summary: 'Moderate content (approve, reject, flag, unflag)' })
  @ApiResponse({ status: 200, description: 'Content moderated successfully' })
  async moderateContent(@Body() request: ContentModerationRequest) {
    await this.adminService.moderateContent(request);
    return { message: 'Content moderated successfully' };
  }

  @Post('ads/manage')
  @ApiOperation({ summary: 'Manage ad (approve, reject, pause, resume)' })
  @ApiResponse({ status: 200, description: 'Ad managed successfully' })
  async manageAd(@Body() request: AdManagementRequest) {
    await this.adminService.manageAd(request);
    return { message: 'Ad managed successfully' };
  }

  @Get('health')
  @ApiOperation({ summary: 'Get system health' })
  @ApiResponse({ status: 200, description: 'System health retrieved successfully' })
  async getSystemHealth() {
    return this.adminService.getSystemHealth();
  }

  @Get('users')
  @ApiOperation({ summary: 'Get users' })
  @ApiResponse({ status: 200, description: 'Users retrieved successfully' })
  async getUsers(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.adminService.getUsers(page, limit);
  }

  @Get('videos')
  @ApiOperation({ summary: 'Get videos' })
  @ApiResponse({ status: 200, description: 'Videos retrieved successfully' })
  async getVideos(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.adminService.getVideos(page, limit);
  }

  @Get('posts')
  @ApiOperation({ summary: 'Get posts' })
  @ApiResponse({ status: 200, description: 'Posts retrieved successfully' })
  async getPosts(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.adminService.getPosts(page, limit);
  }

  @Get('ads')
  @ApiOperation({ summary: 'Get ads' })
  @ApiResponse({ status: 200, description: 'Ads retrieved successfully' })
  async getAds(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.adminService.getAds(page, limit);
  }

  @Get('payments')
  @ApiOperation({ summary: 'Get payments' })
  @ApiResponse({ status: 200, description: 'Payments retrieved successfully' })
  async getPayments(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.adminService.getPayments(page, limit);
  }

  @Get('subscriptions')
  @ApiOperation({ summary: 'Get subscriptions' })
  @ApiResponse({ status: 200, description: 'Subscriptions retrieved successfully' })
  async getSubscriptions(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.adminService.getSubscriptions(page, limit);
  }

  @Get('notifications')
  @ApiOperation({ summary: 'Get notifications' })
  @ApiResponse({ status: 200, description: 'Notifications retrieved successfully' })
  async getNotifications(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.adminService.getNotifications(page, limit);
  }

  @Get('analytics')
  @ApiOperation({ summary: 'Get analytics' })
  @ApiResponse({ status: 200, description: 'Analytics retrieved successfully' })
  async getAnalytics(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.adminService.getAnalytics(page, limit);
  }

  @Get('view-counts')
  @ApiOperation({ summary: 'Get view counts' })
  @ApiResponse({ status: 200, description: 'View counts retrieved successfully' })
  async getViewCounts(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.adminService.getViewCounts(page, limit);
  }
}
