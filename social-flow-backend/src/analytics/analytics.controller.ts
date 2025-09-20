import { Controller, Get, Post, Body, Query, Param, UseGuards, Request } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AnalyticsService, TrackEventRequest, GetAnalyticsRequest } from './analytics.service';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiQuery } from '@nestjs/swagger';

@ApiTags('analytics')
@Controller('analytics')
export class AnalyticsController {
  constructor(private analyticsService: AnalyticsService) {}

  @Post('track')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Track an analytics event' })
  @ApiResponse({ status: 201, description: 'Event tracked successfully' })
  async trackEvent(@Request() req, @Body() eventData: TrackEventRequest) {
    await this.analyticsService.trackEvent(req.user.id, eventData);
    return { message: 'Event tracked successfully' };
  }

  @Post('track/page-view')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Track a page view' })
  @ApiResponse({ status: 201, description: 'Page view tracked successfully' })
  async trackPageView(
    @Request() req,
    @Body() body: { page: string; properties?: Record<string, any> },
  ) {
    await this.analyticsService.trackPageView(req.user.id, body.page, body.properties);
    return { message: 'Page view tracked successfully' };
  }

  @Post('track/video-view')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Track a video view' })
  @ApiResponse({ status: 201, description: 'Video view tracked successfully' })
  async trackVideoView(
    @Request() req,
    @Body() body: { videoId: string; properties?: Record<string, any> },
  ) {
    await this.analyticsService.trackVideoView(req.user.id, body.videoId, body.properties);
    return { message: 'Video view tracked successfully' };
  }

  @Post('track/post-view')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Track a post view' })
  @ApiResponse({ status: 201, description: 'Post view tracked successfully' })
  async trackPostView(
    @Request() req,
    @Body() body: { postId: string; properties?: Record<string, any> },
  ) {
    await this.analyticsService.trackPostView(req.user.id, body.postId, body.properties);
    return { message: 'Post view tracked successfully' };
  }

  @Post('track/user-action')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Track a user action' })
  @ApiResponse({ status: 201, description: 'User action tracked successfully' })
  async trackUserAction(
    @Request() req,
    @Body() body: { action: string; properties?: Record<string, any> },
  ) {
    await this.analyticsService.trackUserAction(req.user.id, body.action, body.properties);
    return { message: 'User action tracked successfully' };
  }

  @Get('user')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get user analytics' })
  @ApiResponse({ status: 200, description: 'User analytics retrieved successfully' })
  async getUserAnalytics(
    @Request() req,
    @Query() query: { startDate: string; endDate: string; groupBy?: string },
  ) {
    const request: GetAnalyticsRequest = {
      startDate: new Date(query.startDate),
      endDate: new Date(query.endDate),
      groupBy: query.groupBy,
    };
    return this.analyticsService.getUserAnalytics(req.user.id, request);
  }

  @Get('content/:entityType/:entityId')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get content analytics' })
  @ApiResponse({ status: 200, description: 'Content analytics retrieved successfully' })
  async getContentAnalytics(
    @Param('entityType') entityType: string,
    @Param('entityId') entityId: string,
    @Query() query: { startDate: string; endDate: string; groupBy?: string },
  ) {
    const request: GetAnalyticsRequest = {
      startDate: new Date(query.startDate),
      endDate: new Date(query.endDate),
      groupBy: query.groupBy,
    };
    return this.analyticsService.getContentAnalytics(entityType, entityId, request);
  }

  @Get('system')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get system analytics' })
  @ApiResponse({ status: 200, description: 'System analytics retrieved successfully' })
  async getSystemAnalytics(
    @Query() query: { startDate: string; endDate: string; groupBy?: string },
  ) {
    const request: GetAnalyticsRequest = {
      startDate: new Date(query.startDate),
      endDate: new Date(query.endDate),
      groupBy: query.groupBy,
    };
    return this.analyticsService.getSystemAnalytics(request);
  }

  @Get('metrics/aggregated')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get aggregated metrics' })
  @ApiResponse({ status: 200, description: 'Aggregated metrics retrieved successfully' })
  async getAggregatedMetrics(
    @Query() query: { startDate: string; endDate: string; groupBy?: string },
  ) {
    const request: GetAnalyticsRequest = {
      startDate: new Date(query.startDate),
      endDate: new Date(query.endDate),
      groupBy: query.groupBy,
    };
    return this.analyticsService.getAggregatedMetrics(request);
  }

  @Get('top/events')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get top events' })
  @ApiResponse({ status: 200, description: 'Top events retrieved successfully' })
  async getTopEvents(@Query('limit') limit: number = 10) {
    return this.analyticsService.getTopEvents(limit);
  }

  @Get('top/users')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get top users' })
  @ApiResponse({ status: 200, description: 'Top users retrieved successfully' })
  async getTopUsers(@Query('limit') limit: number = 10) {
    return this.analyticsService.getTopUsers(limit);
  }

  @Get('top/entities/:entityType')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get top entities' })
  @ApiResponse({ status: 200, description: 'Top entities retrieved successfully' })
  async getTopEntities(
    @Param('entityType') entityType: string,
    @Query('limit') limit: number = 10,
  ) {
    return this.analyticsService.getTopEntities(entityType, limit);
  }

  @Get('overview')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get analytics overview' })
  @ApiResponse({ status: 200, description: 'Analytics overview retrieved successfully' })
  async getAnalyticsOverview() {
    return this.analyticsService.getAnalyticsOverview();
  }

  @Get('video/:videoId')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get video analytics' })
  @ApiResponse({ status: 200, description: 'Video analytics retrieved successfully' })
  async getVideoAnalytics(
    @Param('videoId') videoId: string,
    @Query() query: { startDate: string; endDate: string; groupBy?: string },
  ) {
    const request: GetAnalyticsRequest = {
      startDate: new Date(query.startDate),
      endDate: new Date(query.endDate),
      groupBy: query.groupBy,
    };
    return this.analyticsService.getVideoAnalytics(videoId, request);
  }

  @Get('user/overview')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get user analytics overview' })
  @ApiResponse({ status: 200, description: 'User analytics overview retrieved successfully' })
  async getUserAnalyticsOverview(@Request() req) {
    return this.analyticsService.getUserAnalyticsOverview(req.user.id);
  }

  @Get('content/overview')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get content analytics overview' })
  @ApiResponse({ status: 200, description: 'Content analytics overview retrieved successfully' })
  async getContentAnalyticsOverview() {
    return this.analyticsService.getContentAnalyticsOverview();
  }

  @Get('revenue')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get revenue analytics' })
  @ApiResponse({ status: 200, description: 'Revenue analytics retrieved successfully' })
  async getRevenueAnalytics(
    @Query() query: { startDate: string; endDate: string; groupBy?: string },
  ) {
    const request: GetAnalyticsRequest = {
      startDate: new Date(query.startDate),
      endDate: new Date(query.endDate),
      groupBy: query.groupBy,
    };
    return this.analyticsService.getRevenueAnalytics(request);
  }

  @Get('engagement')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get engagement analytics' })
  @ApiResponse({ status: 200, description: 'Engagement analytics retrieved successfully' })
  async getEngagementAnalytics(
    @Query() query: { startDate: string; endDate: string; groupBy?: string },
  ) {
    const request: GetAnalyticsRequest = {
      startDate: new Date(query.startDate),
      endDate: new Date(query.endDate),
      groupBy: query.groupBy,
    };
    return this.analyticsService.getEngagementAnalytics(request);
  }

  @Post('cleanup')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Cleanup old analytics' })
  @ApiResponse({ status: 200, description: 'Old analytics cleaned up successfully' })
  async cleanupOldAnalytics(@Body() body: { days: number }) {
    await this.analyticsService.cleanupOldAnalytics(body.days);
    return { message: 'Old analytics cleaned up successfully' };
  }
}
