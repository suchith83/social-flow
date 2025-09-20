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

import { NotificationsService, CreateNotificationRequest, SendNotificationRequest } from './notifications.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('notifications')
@Controller('notifications')
@UseGuards(ThrottlerGuard)
export class NotificationsController {
  constructor(private readonly notificationsService: NotificationsService) {}

  @Post()
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Create a new notification' })
  @ApiResponse({ status: 201, description: 'Notification created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  async createNotification(
    @Body(ValidationPipe) notificationData: CreateNotificationRequest,
    @Request() req,
  ) {
    return this.notificationsService.createNotification(req.user.id, notificationData);
  }

  @Post('send')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Send notification to user' })
  @ApiResponse({ status: 201, description: 'Notification sent successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  async sendNotification(
    @Body(ValidationPipe) notificationData: SendNotificationRequest,
  ) {
    return this.notificationsService.sendNotification(notificationData);
  }

  @Get(':id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get notification by ID' })
  @ApiResponse({ status: 200, description: 'Notification retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Notification not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to view this notification' })
  async getNotification(@Param('id') notificationId: string, @Request() req) {
    return this.notificationsService.getNotification(notificationId, req.user.id);
  }

  @Get()
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get user notifications' })
  @ApiResponse({ status: 200, description: 'Notifications retrieved successfully' })
  async getUserNotifications(
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
    @Request() req,
  ) {
    return this.notificationsService.getUserNotifications(req.user.id, limit, offset);
  }

  @Get('unread')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get unread notifications' })
  @ApiResponse({ status: 200, description: 'Unread notifications retrieved successfully' })
  async getUnreadNotifications(
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
    @Request() req,
  ) {
    return this.notificationsService.getUnreadNotifications(req.user.id, limit, offset);
  }

  @Put(':id/read')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Mark notification as read' })
  @ApiResponse({ status: 200, description: 'Notification marked as read' })
  @ApiResponse({ status: 404, description: 'Notification not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to mark this notification as read' })
  async markAsRead(@Param('id') notificationId: string, @Request() req) {
    return this.notificationsService.markAsRead(notificationId, req.user.id);
  }

  @Put('read-all')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Mark all notifications as read' })
  @ApiResponse({ status: 200, description: 'All notifications marked as read' })
  async markAllAsRead(@Request() req) {
    return this.notificationsService.markAllAsRead(req.user.id);
  }

  @Delete(':id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Delete notification' })
  @ApiResponse({ status: 200, description: 'Notification deleted successfully' })
  @ApiResponse({ status: 404, description: 'Notification not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to delete this notification' })
  async deleteNotification(@Param('id') notificationId: string, @Request() req) {
    return this.notificationsService.deleteNotification(notificationId, req.user.id);
  }

  @Get('stats/count')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get unread notification count' })
  @ApiResponse({ status: 200, description: 'Unread count retrieved successfully' })
  async getUnreadCount(@Request() req) {
    const count = await this.notificationsService.getUnreadCount(req.user.id);
    return { count };
  }

  @Get('stats/overview')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get notification statistics' })
  @ApiResponse({ status: 200, description: 'Notification stats retrieved successfully' })
  async getNotificationStats(@Request() req) {
    return this.notificationsService.getNotificationStats(req.user.id);
  }

  // System notification endpoints (Admin only)
  @Post('system/announcement')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Send system announcement (Admin only)' })
  @ApiResponse({ status: 200, description: 'System announcement sent successfully' })
  @ApiResponse({ status: 403, description: 'Not authorized to send system announcements' })
  async sendSystemAnnouncement(
    @Body('title') title: string,
    @Body('message') message: string,
    @Body('actionUrl') actionUrl?: string,
    @Body('imageUrl') imageUrl?: string,
    @Body('targetUsers') targetUsers?: string[],
    @Request() req,
  ) {
    // Check if user is admin
    if (!req.user.isAdmin) {
      throw new ForbiddenException('Not authorized to send system announcements');
    }

    return this.notificationsService.sendSystemAnnouncement(
      title,
      message,
      actionUrl,
      imageUrl,
      targetUsers,
    );
  }

  @Post('system/security-alert')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Send security alert (Admin only)' })
  @ApiResponse({ status: 200, description: 'Security alert sent successfully' })
  @ApiResponse({ status: 403, description: 'Not authorized to send security alerts' })
  async sendSecurityAlert(
    @Body('userId') userId: string,
    @Body('title') title: string,
    @Body('message') message: string,
    @Body('actionUrl') actionUrl?: string,
    @Request() req,
  ) {
    // Check if user is admin
    if (!req.user.isAdmin) {
      throw new ForbiddenException('Not authorized to send security alerts');
    }

    return this.notificationsService.sendSecurityAlert(userId, title, message, actionUrl);
  }
}
