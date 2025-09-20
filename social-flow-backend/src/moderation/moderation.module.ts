import { Module } from '@nestjs/common';
import { DatabaseModule } from '../shared/database/database.module';
import { AwsModule } from '../shared/aws/aws.module';
import { ModerationService } from './moderation.service';
import { ModerationController } from './moderation.controller';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { VideoRepository } from '../shared/database/repositories/video.repository';
import { PostRepository } from '../shared/database/repositories/post.repository';
import { CommentRepository } from '../shared/database/repositories/comment.repository';

@Module({
  imports: [DatabaseModule, AwsModule],
  providers: [
    ModerationService,
    UserRepository,
    VideoRepository,
    PostRepository,
    CommentRepository,
  ],
  controllers: [ModerationController],
})
export class ModerationModule {}
