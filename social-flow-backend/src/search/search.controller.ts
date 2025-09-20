import { Controller, Get, Post, Body, Query, UseGuards, Request } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { SearchService, SearchRequest } from './search.service';
import { RecommendationService, RecommendationRequest } from './recommendation.service';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiQuery } from '@nestjs/swagger';

@ApiTags('search')
@Controller('search')
export class SearchController {
  constructor(
    private searchService: SearchService,
    private recommendationService: RecommendationService,
  ) {}

  @Post()
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Search content' })
  @ApiResponse({ status: 200, description: 'Search results retrieved successfully' })
  async search(@Body() request: SearchRequest) {
    return this.searchService.search(request);
  }

  @Get('suggestions')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get search suggestions' })
  @ApiResponse({ status: 200, description: 'Search suggestions retrieved successfully' })
  async getSuggestions(
    @Query('q') query: string,
    @Query('type') type?: string,
  ) {
    return this.searchService.getSuggestions(query, type);
  }

  @Get('trending/hashtags')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get trending hashtags' })
  @ApiResponse({ status: 200, description: 'Trending hashtags retrieved successfully' })
  async getTrendingHashtags(@Query('limit') limit: number = 10) {
    return this.searchService.getTrendingHashtags(limit);
  }

  @Get('trending/topics')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get trending topics' })
  @ApiResponse({ status: 200, description: 'Trending topics retrieved successfully' })
  async getTrendingTopics(@Query('limit') limit: number = 10) {
    return this.searchService.getTrendingTopics(limit);
  }

  @Post('recommendations')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get recommendations' })
  @ApiResponse({ status: 200, description: 'Recommendations retrieved successfully' })
  async getRecommendations(@Body() request: RecommendationRequest) {
    return this.recommendationService.getRecommendations(request);
  }

  @Get('trending/videos')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get trending videos' })
  @ApiResponse({ status: 200, description: 'Trending videos retrieved successfully' })
  async getTrendingVideos(@Query('limit') limit: number = 20) {
    return this.recommendationService.getTrendingVideos(limit);
  }

  @Get('trending/posts')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get trending posts' })
  @ApiResponse({ status: 200, description: 'Trending posts retrieved successfully' })
  async getTrendingPosts(@Query('limit') limit: number = 20) {
    return this.recommendationService.getTrendingPosts(limit);
  }

  @Get('trending/users')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get trending users' })
  @ApiResponse({ status: 200, description: 'Trending users retrieved successfully' })
  async getTrendingUsers(@Query('limit') limit: number = 20) {
    return this.recommendationService.getTrendingUsers(limit);
  }
}
