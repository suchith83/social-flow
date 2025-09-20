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

import { AdsService, CreateAdRequest, UpdateAdRequest } from './ads.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('ads')
@Controller('ads')
@UseGuards(ThrottlerGuard)
export class AdsController {
  constructor(private readonly adsService: AdsService) {}

  @Post()
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Create a new ad' })
  @ApiResponse({ status: 201, description: 'Ad created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  async createAd(
    @Body(ValidationPipe) adData: CreateAdRequest,
    @Request() req,
  ) {
    return this.adsService.createAd(req.user.id, adData);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get ad by ID' })
  @ApiResponse({ status: 200, description: 'Ad retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  async getAd(@Param('id') adId: string) {
    return this.adsService.getAd(adId);
  }

  @Put(':id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update ad' })
  @ApiResponse({ status: 200, description: 'Ad updated successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to update this ad' })
  async updateAd(
    @Param('id') adId: string,
    @Body(ValidationPipe) updateData: UpdateAdRequest,
    @Request() req,
  ) {
    return this.adsService.updateAd(adId, req.user.id, updateData);
  }

  @Delete(':id')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Delete ad' })
  @ApiResponse({ status: 200, description: 'Ad deleted successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to delete this ad' })
  async deleteAd(@Param('id') adId: string, @Request() req) {
    return this.adsService.deleteAd(adId, req.user.id);
  }

  @Post(':id/approve')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Approve ad (Admin/Moderator only)' })
  @ApiResponse({ status: 200, description: 'Ad approved successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to approve ads' })
  async approveAd(@Param('id') adId: string, @Request() req) {
    return this.adsService.approveAd(adId, req.user.id);
  }

  @Post(':id/reject')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Reject ad (Admin/Moderator only)' })
  @ApiResponse({ status: 200, description: 'Ad rejected successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to reject ads' })
  async rejectAd(
    @Param('id') adId: string,
    @Body('reason') reason: string,
    @Request() req,
  ) {
    return this.adsService.rejectAd(adId, req.user.id, reason);
  }

  @Get()
  @ApiOperation({ summary: 'Get active ads' })
  @ApiResponse({ status: 200, description: 'Ads retrieved successfully' })
  async getActiveAds(
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.adsService.getActiveAds(limit, offset);
  }

  @Get('advertiser/:advertiserId')
  @ApiOperation({ summary: 'Get ads by advertiser' })
  @ApiResponse({ status: 200, description: 'Ads retrieved successfully' })
  async getAdsByAdvertiser(
    @Param('advertiserId') advertiserId: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.adsService.getAdsByAdvertiser(advertiserId, limit, offset);
  }

  @Get('type/:type')
  @ApiOperation({ summary: 'Get ads by type' })
  @ApiResponse({ status: 200, description: 'Ads retrieved successfully' })
  async getAdsByType(
    @Param('type') type: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.adsService.getAdsByType(type as any, limit, offset);
  }

  @Post(':id/impression')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Record ad impression' })
  @ApiResponse({ status: 200, description: 'Impression recorded successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  async recordImpression(@Param('id') adId: string) {
    return this.adsService.recordImpression(adId);
  }

  @Post(':id/click')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Record ad click' })
  @ApiResponse({ status: 200, description: 'Click recorded successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  async recordClick(@Param('id') adId: string) {
    return this.adsService.recordClick(adId);
  }

  @Post(':id/conversion')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Record ad conversion' })
  @ApiResponse({ status: 200, description: 'Conversion recorded successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  async recordConversion(@Param('id') adId: string) {
    return this.adsService.recordConversion(adId);
  }

  @Get(':id/stats')
  @ApiOperation({ summary: 'Get ad statistics' })
  @ApiResponse({ status: 200, description: 'Stats retrieved successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  async getAdStats(@Param('id') adId: string) {
    return this.adsService.getAdStats(adId);
  }

  @Post(':id/pause')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Pause ad' })
  @ApiResponse({ status: 200, description: 'Ad paused successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to pause this ad' })
  async pauseAd(@Param('id') adId: string, @Request() req) {
    return this.adsService.pauseAd(adId, req.user.id);
  }

  @Post(':id/resume')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Resume ad' })
  @ApiResponse({ status: 200, description: 'Ad resumed successfully' })
  @ApiResponse({ status: 404, description: 'Ad not found' })
  @ApiResponse({ status: 403, description: 'Not authorized to resume this ad' })
  async resumeAd(@Param('id') adId: string, @Request() req) {
    return this.adsService.resumeAd(adId, req.user.id);
  }
}
