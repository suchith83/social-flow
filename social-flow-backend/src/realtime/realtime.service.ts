import { Injectable } from '@nestjs/common';
import { RedisService } from '../shared/redis/redis.service';
import { LoggerService } from '../shared/logger/logger.service';

export interface RealtimeEvent {
  type: string;
  data: any;
  userId?: string;
  roomId?: string;
}

@Injectable()
export class RealtimeService {
  constructor(
    private redisService: RedisService,
    private logger: LoggerService,
  ) {}

  async publishEvent(event: RealtimeEvent): Promise<void> {
    try {
      const channel = event.roomId ? `room:${event.roomId}` : 'global';
      await this.redisService.publish(channel, JSON.stringify(event));
      
      this.logger.logBusiness('realtime_event_published', event.userId, {
        type: event.type,
        roomId: event.roomId,
      });
    } catch (error) {
      this.logger.logError(error, 'RealtimeService.publishEvent', { event });
      throw error;
    }
  }

  async subscribeToRoom(roomId: string, callback: (event: RealtimeEvent) => void): Promise<void> {
    try {
      const channel = `room:${roomId}`;
      await this.redisService.subscribe(channel, (message) => {
        try {
          const event = JSON.parse(message);
          callback(event);
        } catch (error) {
          this.logger.logError(error, 'RealtimeService.subscribeToRoom', { roomId, message });
        }
      });
    } catch (error) {
      this.logger.logError(error, 'RealtimeService.subscribeToRoom', { roomId });
      throw error;
    }
  }

  async subscribeToUser(userId: string, callback: (event: RealtimeEvent) => void): Promise<void> {
    try {
      const channel = `user:${userId}`;
      await this.redisService.subscribe(channel, (message) => {
        try {
          const event = JSON.parse(message);
          callback(event);
        } catch (error) {
          this.logger.logError(error, 'RealtimeService.subscribeToUser', { userId, message });
        }
      });
    } catch (error) {
      this.logger.logError(error, 'RealtimeService.subscribeToUser', { userId });
      throw error;
    }
  }

  async subscribeToGlobal(callback: (event: RealtimeEvent) => void): Promise<void> {
    try {
      const channel = 'global';
      await this.redisService.subscribe(channel, (message) => {
        try {
          const event = JSON.parse(message);
          callback(event);
        } catch (error) {
          this.logger.logError(error, 'RealtimeService.subscribeToGlobal', { message });
        }
      });
    } catch (error) {
      this.logger.logError(error, 'RealtimeService.subscribeToGlobal');
      throw error;
    }
  }

  async unsubscribeFromRoom(roomId: string): Promise<void> {
    try {
      const channel = `room:${roomId}`;
      await this.redisService.unsubscribe(channel);
    } catch (error) {
      this.logger.logError(error, 'RealtimeService.unsubscribeFromRoom', { roomId });
      throw error;
    }
  }

  async unsubscribeFromUser(userId: string): Promise<void> {
    try {
      const channel = `user:${userId}`;
      await this.redisService.unsubscribe(channel);
    } catch (error) {
      this.logger.logError(error, 'RealtimeService.unsubscribeFromUser', { userId });
      throw error;
    }
  }

  async unsubscribeFromGlobal(): Promise<void> {
    try {
      const channel = 'global';
      await this.redisService.unsubscribe(channel);
    } catch (error) {
      this.logger.logError(error, 'RealtimeService.unsubscribeFromGlobal');
      throw error;
    }
  }
}
