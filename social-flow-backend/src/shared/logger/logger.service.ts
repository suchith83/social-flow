import { Injectable, LoggerService as NestLoggerService } from '@nestjs/common';
import * as winston from 'winston';

@Injectable()
export class LoggerService implements NestLoggerService {
  private readonly logger: winston.Logger;

  constructor() {
    this.logger = winston.createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json(),
      ),
      defaultMeta: { service: 'social-flow-backend' },
      transports: [
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.simple(),
          ),
        }),
        new winston.transports.File({
          filename: 'logs/error.log',
          level: 'error',
        }),
        new winston.transports.File({
          filename: 'logs/combined.log',
        }),
      ],
    });
  }

  log(message: any, context?: string) {
    this.logger.info(message, { context });
  }

  error(message: any, trace?: string, context?: string) {
    this.logger.error(message, { trace, context });
  }

  warn(message: any, context?: string) {
    this.logger.warn(message, { context });
  }

  debug(message: any, context?: string) {
    this.logger.debug(message, { context });
  }

  verbose(message: any, context?: string) {
    this.logger.verbose(message, { context });
  }

  // Custom logging methods
  logRequest(method: string, url: string, statusCode: number, responseTime: number, userAgent?: string) {
    this.logger.info('HTTP Request', {
      method,
      url,
      statusCode,
      responseTime,
      userAgent,
    });
  }

  logError(error: Error, context?: string, metadata?: Record<string, any>) {
    this.logger.error(error.message, {
      stack: error.stack,
      context,
      ...metadata,
    });
  }

  logSecurity(event: string, userId?: string, ip?: string, metadata?: Record<string, any>) {
    this.logger.warn('Security Event', {
      event,
      userId,
      ip,
      ...metadata,
    });
  }

  logBusiness(event: string, userId?: string, metadata?: Record<string, any>) {
    this.logger.info('Business Event', {
      event,
      userId,
      ...metadata,
    });
  }

  logPerformance(operation: string, duration: number, metadata?: Record<string, any>) {
    this.logger.info('Performance', {
      operation,
      duration,
      ...metadata,
    });
  }
}
