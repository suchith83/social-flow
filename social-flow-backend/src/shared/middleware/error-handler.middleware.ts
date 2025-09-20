import { Injectable, NestMiddleware, HttpException, HttpStatus } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import { LoggerService } from '../logger/logger.service';

@Injectable()
export class ErrorHandlerMiddleware implements NestMiddleware {
  constructor(private logger: LoggerService) {}

  use(req: Request, res: Response, next: NextFunction) {
    const originalSend = res.send;
    
    res.send = function(data) {
      try {
        const parsed = JSON.parse(data);
        if (parsed.statusCode >= 400) {
          this.logger.logError(new Error(parsed.message), 'ErrorHandlerMiddleware', {
            url: req.url,
            method: req.method,
            statusCode: parsed.statusCode,
          });
        }
      } catch (e) {
        // Not JSON, ignore
      }
      
      return originalSend.call(this, data);
    }.bind(res);
    
    next();
  }
}
