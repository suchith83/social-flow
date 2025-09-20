import { Controller, Get, Post, Body, Query, UseGuards, Request } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ModerationService, ContentModerationRequest } from './moderation.service';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiQuery } from '@nestjs/swagger';

@ApiTags('moderation')
@Controller('moderation')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class ModerationController {
  constructor(private moderationService: ModerationService) {}

  @Post('moderate')
  @ApiOperation({ summary: 'Moderate content' })
  @ApiResponse({ status: 200, description: 'Content moderated successfully' })
  async moderateContent(@Body() request: ContentModerationRequest) {
    return this.moderationService.moderateContent(request);
  }

  @Get('flagged')
  @ApiOperation({ summary: 'Get flagged content' })
  @ApiResponse({ status: 200, description: 'Flagged content retrieved successfully' })
  async getFlaggedContent(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.moderationService.getFlaggedContent(page, limit);
  }

  @Get('queue')
  @ApiOperation({ summary: 'Get moderation queue' })
  @ApiResponse({ status: 200, description: 'Moderation queue retrieved successfully' })
  async getModerationQueue(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 20,
  ) {
    return this.moderationService.getModerationQueue(page, limit);
  }

  @Get('stats')
  @ApiOperation({ summary: 'Get moderation statistics' })
  @ApiResponse({ status: 200, description: 'Moderation statistics retrieved successfully' })
  async getModerationStats() {
    return this.moderationService.getModerationStats();
  }
}
