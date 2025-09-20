import { Injectable, NestMiddleware, HttpException, HttpStatus } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import { RedisService } from '../redis/redis.service';

@Injectable()
export class RateLimitMiddleware implements NestMiddleware {
  constructor(private redisService: RedisService) {}

  async use(req: Request, res: Response, next: NextFunction) {
    const ip = req.ip || req.connection.remoteAddress;
    const key = `rate_limit:${ip}`;
    
    try {
      const current = await this.redisService.get(key);
      const limit = 100; // requests per minute
      const window = 60; // seconds
      
      if (current === null) {
        await this.redisService.setex(key, window, '1');
      } else {
        const count = parseInt(current);
        if (count >= limit) {
          throw new HttpException('Too Many Requests', HttpStatus.TOO_MANY_REQUESTS);
        }
        await this.redisService.incr(key);
      }
      
      next();
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      next();
    }
  }
}
