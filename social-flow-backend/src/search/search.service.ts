import { Injectable, NotFoundException } from '@nestjs/common';
import { ElasticsearchService } from '../shared/aws/elasticsearch.service';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { VideoRepository } from '../shared/database/repositories/video.repository';
import { PostRepository } from '../shared/database/repositories/post.repository';
import { LoggerService } from '../shared/logger/logger.service';

export interface SearchRequest {
  query: string;
  type?: 'all' | 'users' | 'videos' | 'posts';
  page?: number;
  limit?: number;
  filters?: Record<string, any>;
  sort?: string;
  order?: 'asc' | 'desc';
}

export interface SearchResult {
  results: any[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

@Injectable()
export class SearchService {
  constructor(
    private elasticsearchService: ElasticsearchService,
    private userRepository: UserRepository,
    private videoRepository: VideoRepository,
    private postRepository: PostRepository,
    private logger: LoggerService,
  ) {}

  async search(request: SearchRequest): Promise<SearchResult> {
    try {
      const { query, type = 'all', page = 1, limit = 20, filters = {}, sort = 'relevance', order = 'desc' } = request;
      
      const offset = (page - 1) * limit;
      
      let results: any[] = [];
      let total = 0;
      
      switch (type) {
        case 'users':
          const userResults = await this.searchUsers(query, offset, limit, filters, sort, order);
          results = userResults.results;
          total = userResults.total;
          break;
        case 'videos':
          const videoResults = await this.searchVideos(query, offset, limit, filters, sort, order);
          results = videoResults.results;
          total = videoResults.total;
          break;
        case 'posts':
          const postResults = await this.searchPosts(query, offset, limit, filters, sort, order);
          results = postResults.results;
          total = postResults.total;
          break;
        case 'all':
        default:
          const allResults = await this.searchAll(query, offset, limit, filters, sort, order);
          results = allResults.results;
          total = allResults.total;
          break;
      }
      
      const totalPages = Math.ceil(total / limit);
      
      return {
        results,
        total,
        page,
        limit,
        totalPages,
      };
    } catch (error) {
      this.logger.logError(error, 'SearchService.search', { request });
      throw error;
    }
  }

  async searchUsers(query: string, offset: number, limit: number, filters: Record<string, any>, sort: string, order: 'asc' | 'desc'): Promise<{ results: any[]; total: number }> {
    try {
      const searchBody = {
        query: {
          bool: {
            must: [
              {
                multi_match: {
                  query,
                  fields: ['username^2', 'displayName^1.5', 'bio'],
                  type: 'best_fields',
                },
              },
            ],
            filter: this.buildFilters(filters),
          },
        },
        sort: this.buildSort(sort, order),
        from: offset,
        size: limit,
      };
      
      const response = await this.elasticsearchService.search('users', searchBody);
      
      return {
        results: response.hits.hits.map(hit => hit._source),
        total: response.hits.total.value,
      };
    } catch (error) {
      this.logger.logError(error, 'SearchService.searchUsers', { query, offset, limit, filters, sort, order });
      throw error;
    }
  }

  async searchVideos(query: string, offset: number, limit: number, filters: Record<string, any>, sort: string, order: 'asc' | 'desc'): Promise<{ results: any[]; total: number }> {
    try {
      const searchBody = {
        query: {
          bool: {
            must: [
              {
                multi_match: {
                  query,
                  fields: ['title^2', 'description^1.5', 'tags'],
                  type: 'best_fields',
                },
              },
            ],
            filter: this.buildFilters(filters),
          },
        },
        sort: this.buildSort(sort, order),
        from: offset,
        size: limit,
      };
      
      const response = await this.elasticsearchService.search('videos', searchBody);
      
      return {
        results: response.hits.hits.map(hit => hit._source),
        total: response.hits.total.value,
      };
    } catch (error) {
      this.logger.logError(error, 'SearchService.searchVideos', { query, offset, limit, filters, sort, order });
      throw error;
    }
  }

  async searchPosts(query: string, offset: number, limit: number, filters: Record<string, any>, sort: string, order: 'asc' | 'desc'): Promise<{ results: any[]; total: number }> {
    try {
      const searchBody = {
        query: {
          bool: {
            must: [
              {
                multi_match: {
                  query,
                  fields: ['content^2', 'hashtags'],
                  type: 'best_fields',
                },
              },
            ],
            filter: this.buildFilters(filters),
          },
        },
        sort: this.buildSort(sort, order),
        from: offset,
        size: limit,
      };
      
      const response = await this.elasticsearchService.search('posts', searchBody);
      
      return {
        results: response.hits.hits.map(hit => hit._source),
        total: response.hits.total.value,
      };
    } catch (error) {
      this.logger.logError(error, 'SearchService.searchPosts', { query, offset, limit, filters, sort, order });
      throw error;
    }
  }

  async searchAll(query: string, offset: number, limit: number, filters: Record<string, any>, sort: string, order: 'asc' | 'desc'): Promise<{ results: any[]; total: number }> {
    try {
      const searchBody = {
        query: {
          bool: {
            must: [
              {
                multi_match: {
                  query,
                  fields: ['username^2', 'displayName^1.5', 'bio', 'title^2', 'description^1.5', 'tags', 'content^2', 'hashtags'],
                  type: 'best_fields',
                },
              },
            ],
            filter: this.buildFilters(filters),
          },
        },
        sort: this.buildSort(sort, order),
        from: offset,
        size: limit,
      };
      
      const response = await this.elasticsearchService.search('_all', searchBody);
      
      return {
        results: response.hits.hits.map(hit => hit._source),
        total: response.hits.total.value,
      };
    } catch (error) {
      this.logger.logError(error, 'SearchService.searchAll', { query, offset, limit, filters, sort, order });
      throw error;
    }
  }

  async getSuggestions(query: string, type?: string): Promise<string[]> {
    try {
      const searchBody = {
        suggest: {
          suggestion: {
            prefix: query,
            completion: {
              field: 'suggest',
              size: 10,
            },
          },
        },
      };
      
      const response = await this.elasticsearchService.search(type || '_all', searchBody);
      
      return response.suggest.suggestion[0].options.map(option => option.text);
    } catch (error) {
      this.logger.logError(error, 'SearchService.getSuggestions', { query, type });
      throw error;
    }
  }

  async getTrendingHashtags(limit: number = 10): Promise<string[]> {
    try {
      const searchBody = {
        aggs: {
          hashtags: {
            terms: {
              field: 'hashtags.keyword',
              size: limit,
            },
          },
        },
        size: 0,
      };
      
      const response = await this.elasticsearchService.search('posts', searchBody);
      
      return response.aggregations.hashtags.buckets.map(bucket => bucket.key);
    } catch (error) {
      this.logger.logError(error, 'SearchService.getTrendingHashtags', { limit });
      throw error;
    }
  }

  async getTrendingTopics(limit: number = 10): Promise<string[]> {
    try {
      const searchBody = {
        aggs: {
          topics: {
            terms: {
              field: 'tags.keyword',
              size: limit,
            },
          },
        },
        size: 0,
      };
      
      const response = await this.elasticsearchService.search('videos', searchBody);
      
      return response.aggregations.topics.buckets.map(bucket => bucket.key);
    } catch (error) {
      this.logger.logError(error, 'SearchService.getTrendingTopics', { limit });
      throw error;
    }
  }

  private buildFilters(filters: Record<string, any>): any[] {
    const filterArray = [];
    
    Object.keys(filters).forEach(key => {
      if (filters[key] !== undefined && filters[key] !== null) {
        if (Array.isArray(filters[key])) {
          filterArray.push({
            terms: {
              [key]: filters[key],
            },
          });
        } else {
          filterArray.push({
            term: {
              [key]: filters[key],
            },
          });
        }
      }
    });
    
    return filterArray;
  }

  private buildSort(sort: string, order: 'asc' | 'desc'): any[] {
    const sortArray = [];
    
    switch (sort) {
      case 'relevance':
        sortArray.push({ _score: { order } });
        break;
      case 'date':
        sortArray.push({ createdAt: { order } });
        break;
      case 'popularity':
        sortArray.push({ views: { order } });
        break;
      case 'likes':
        sortArray.push({ likes: { order } });
        break;
      default:
        sortArray.push({ _score: { order: 'desc' } });
        break;
    }
    
    return sortArray;
  }
}
