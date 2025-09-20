import { WebSocketGateway, WebSocketServer, SubscribeMessage, OnGatewayConnection, OnGatewayDisconnect, ConnectedSocket, MessageBody } from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { RealtimeService } from './realtime.service';
import { LoggerService } from '../shared/logger/logger.service';

@WebSocketGateway({
  cors: {
    origin: '*',
  },
})
export class RealtimeGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer()
  server: Server;

  constructor(
    private realtimeService: RealtimeService,
    private logger: LoggerService,
  ) {}

  async handleConnection(client: Socket) {
    try {
      const userId = client.handshake.auth.userId;
      if (userId) {
        client.join(`user:${userId}`);
        this.logger.logBusiness('user_connected', userId, { socketId: client.id });
      }
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handleConnection', { socketId: client.id });
    }
  }

  async handleDisconnect(client: Socket) {
    try {
      const userId = client.handshake.auth.userId;
      if (userId) {
        this.logger.logBusiness('user_disconnected', userId, { socketId: client.id });
      }
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handleDisconnect', { socketId: client.id });
    }
  }

  @SubscribeMessage('join_room')
  async handleJoinRoom(@ConnectedSocket() client: Socket, @MessageBody() data: { roomId: string }) {
    try {
      const { roomId } = data;
      client.join(`room:${roomId}`);
      this.logger.logBusiness('user_joined_room', client.handshake.auth.userId, { roomId, socketId: client.id });
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handleJoinRoom', { roomId: data.roomId, socketId: client.id });
    }
  }

  @SubscribeMessage('leave_room')
  async handleLeaveRoom(@ConnectedSocket() client: Socket, @MessageBody() data: { roomId: string }) {
    try {
      const { roomId } = data;
      client.leave(`room:${roomId}`);
      this.logger.logBusiness('user_left_room', client.handshake.auth.userId, { roomId, socketId: client.id });
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handleLeaveRoom', { roomId: data.roomId, socketId: client.id });
    }
  }

  @SubscribeMessage('send_message')
  async handleSendMessage(@ConnectedSocket() client: Socket, @MessageBody() data: { roomId: string; message: string }) {
    try {
      const { roomId, message } = data;
      const userId = client.handshake.auth.userId;
      
      const event = {
        type: 'message',
        data: {
          message,
          userId,
          timestamp: new Date(),
        },
        userId,
        roomId,
      };
      
      await this.realtimeService.publishEvent(event);
      this.logger.logBusiness('message_sent', userId, { roomId, socketId: client.id });
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handleSendMessage', { roomId: data.roomId, socketId: client.id });
    }
  }

  @SubscribeMessage('typing_start')
  async handleTypingStart(@ConnectedSocket() client: Socket, @MessageBody() data: { roomId: string }) {
    try {
      const { roomId } = data;
      const userId = client.handshake.auth.userId;
      
      const event = {
        type: 'typing_start',
        data: { userId },
        userId,
        roomId,
      };
      
      await this.realtimeService.publishEvent(event);
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handleTypingStart', { roomId: data.roomId, socketId: client.id });
    }
  }

  @SubscribeMessage('typing_stop')
  async handleTypingStop(@ConnectedSocket() client: Socket, @MessageBody() data: { roomId: string }) {
    try {
      const { roomId } = data;
      const userId = client.handshake.auth.userId;
      
      const event = {
        type: 'typing_stop',
        data: { userId },
        userId,
        roomId,
      };
      
      await this.realtimeService.publishEvent(event);
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handleTypingStop', { roomId: data.roomId, socketId: client.id });
    }
  }

  @SubscribeMessage('video_view')
  async handleVideoView(@ConnectedSocket() client: Socket, @MessageBody() data: { videoId: string }) {
    try {
      const { videoId } = data;
      const userId = client.handshake.auth.userId;
      
      const event = {
        type: 'video_view',
        data: { videoId, userId },
        userId,
      };
      
      await this.realtimeService.publishEvent(event);
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handleVideoView', { videoId: data.videoId, socketId: client.id });
    }
  }

  @SubscribeMessage('post_like')
  async handlePostLike(@ConnectedSocket() client: Socket, @MessageBody() data: { postId: string }) {
    try {
      const { postId } = data;
      const userId = client.handshake.auth.userId;
      
      const event = {
        type: 'post_like',
        data: { postId, userId },
        userId,
      };
      
      await this.realtimeService.publishEvent(event);
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handlePostLike', { postId: data.postId, socketId: client.id });
    }
  }

  @SubscribeMessage('post_comment')
  async handlePostComment(@ConnectedSocket() client: Socket, @MessageBody() data: { postId: string; comment: string }) {
    try {
      const { postId, comment } = data;
      const userId = client.handshake.auth.userId;
      
      const event = {
        type: 'post_comment',
        data: { postId, comment, userId },
        userId,
      };
      
      await this.realtimeService.publishEvent(event);
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handlePostComment', { postId: data.postId, socketId: client.id });
    }
  }

  @SubscribeMessage('user_follow')
  async handleUserFollow(@ConnectedSocket() client: Socket, @MessageBody() data: { followingId: string }) {
    try {
      const { followingId } = data;
      const userId = client.handshake.auth.userId;
      
      const event = {
        type: 'user_follow',
        data: { followingId, userId },
        userId,
      };
      
      await this.realtimeService.publishEvent(event);
    } catch (error) {
      this.logger.logError(error, 'RealtimeGateway.handleUserFollow', { followingId: data.followingId, socketId: client.id });
    }
  }
}
