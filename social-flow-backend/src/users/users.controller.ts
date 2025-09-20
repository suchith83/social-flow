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
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiConsumes } from '@nestjs/swagger';
import { ThrottlerGuard } from '@nestjs/throttler';

import { UsersService, UpdateProfileRequest, FollowUserRequest } from './users.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('users')
@Controller('users')
@UseGuards(ThrottlerGuard)
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @Get('me')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get current user profile' })
  @ApiResponse({ status: 200, description: 'User profile retrieved successfully' })
  async getCurrentUser(@Request() req) {
    return this.usersService.getProfile(req.user.id);
  }

  @Put('me')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update current user profile' })
  @ApiResponse({ status: 200, description: 'Profile updated successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  async updateProfile(
    @Body(ValidationPipe) updateData: UpdateProfileRequest,
    @Request() req,
  ) {
    return this.usersService.updateProfile(req.user.id, updateData);
  }

  @Post('me/avatar')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiConsumes('multipart/form-data')
  @ApiOperation({ summary: 'Upload user avatar' })
  @ApiResponse({ status: 200, description: 'Avatar uploaded successfully' })
  @UseInterceptors(FileInterceptor('avatar'))
  async uploadAvatar(
    @UploadedFile() file: Express.Multer.File,
    @Request() req,
  ) {
    return this.usersService.uploadAvatar(req.user.id, file);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get user profile by ID' })
  @ApiResponse({ status: 200, description: 'User profile retrieved successfully' })
  @ApiResponse({ status: 404, description: 'User not found' })
  async getUserProfile(@Param('id') userId: string) {
    return this.usersService.getProfile(userId);
  }

  @Post('follow')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Follow a user' })
  @ApiResponse({ status: 200, description: 'User followed successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  @ApiResponse({ status: 403, description: 'Already following or cannot follow yourself' })
  async followUser(
    @Body(ValidationPipe) followData: FollowUserRequest,
    @Request() req,
  ) {
    return this.usersService.followUser({
      ...followData,
      userId: req.user.id,
    });
  }

  @Post('unfollow')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Unfollow a user' })
  @ApiResponse({ status: 200, description: 'User unfollowed successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input data' })
  @ApiResponse({ status: 403, description: 'Not following this user' })
  async unfollowUser(
    @Body(ValidationPipe) followData: FollowUserRequest,
    @Request() req,
  ) {
    return this.usersService.unfollowUser({
      ...followData,
      userId: req.user.id,
    });
  }

  @Get(':id/followers')
  @ApiOperation({ summary: 'Get user followers' })
  @ApiResponse({ status: 200, description: 'Followers retrieved successfully' })
  @ApiResponse({ status: 404, description: 'User not found' })
  async getFollowers(
    @Param('id') userId: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.usersService.getFollowers(userId, limit, offset);
  }

  @Get(':id/following')
  @ApiOperation({ summary: 'Get user following' })
  @ApiResponse({ status: 200, description: 'Following retrieved successfully' })
  @ApiResponse({ status: 404, description: 'User not found' })
  async getFollowing(
    @Param('id') userId: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.usersService.getFollowing(userId, limit, offset);
  }

  @Get('search')
  @ApiOperation({ summary: 'Search users' })
  @ApiResponse({ status: 200, description: 'Users retrieved successfully' })
  async searchUsers(
    @Query('q') query: string,
    @Query('limit') limit: number = 10,
    @Query('offset') offset: number = 0,
  ) {
    return this.usersService.searchUsers(query, limit, offset);
  }

  @Get('top/creators')
  @ApiOperation({ summary: 'Get top creators' })
  @ApiResponse({ status: 200, description: 'Top creators retrieved successfully' })
  async getTopCreators(@Query('limit') limit: number = 10) {
    return this.usersService.getTopCreators(limit);
  }

  @Get('top/followers')
  @ApiOperation({ summary: 'Get users with most followers' })
  @ApiResponse({ status: 200, description: 'Users retrieved successfully' })
  async getUsersWithMostFollowers(@Query('limit') limit: number = 10) {
    return this.usersService.getUsersWithMostFollowers(limit);
  }

  @Get(':id/following-status')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Check if following a user' })
  @ApiResponse({ status: 200, description: 'Following status retrieved successfully' })
  async isFollowing(
    @Param('id') targetUserId: string,
    @Request() req,
  ) {
    const isFollowing = await this.usersService.isFollowing(req.user.id, targetUserId);
    return { isFollowing };
  }

  @Get(':id/stats')
  @ApiOperation({ summary: 'Get user follow stats' })
  @ApiResponse({ status: 200, description: 'Stats retrieved successfully' })
  @ApiResponse({ status: 404, description: 'User not found' })
  async getFollowStats(@Param('id') userId: string) {
    return this.usersService.getFollowStats(userId);
  }

  @Delete('me')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Delete current user account' })
  @ApiResponse({ status: 200, description: 'Account deleted successfully' })
  async deleteUser(@Request() req) {
    return this.usersService.deleteUser(req.user.id);
  }
}
