import { Module } from '@nestjs/common';
import { PostsController } from './posts.controller';
import { PostsService } from './posts.service';
import { CommentsService } from './comments.service';
import { LikesService } from './likes.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Module({
  controllers: [PostsController],
  providers: [PostsService, CommentsService, LikesService, JwtAuthGuard],
  exports: [PostsService, CommentsService, LikesService],
})
export class PostsModule {}
