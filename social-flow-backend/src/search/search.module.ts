import { Module } from '@nestjs/common';
import { DatabaseModule } from '../shared/database/database.module';
import { AwsModule } from '../shared/aws/aws.module';
import { SearchService } from './search.service';
import { SearchController } from './search.controller';
import { RecommendationService } from './recommendation.service';
import { UserRepository } from '../shared/database/repositories/user.repository';
import { VideoRepository } from '../shared/database/repositories/video.repository';
import { PostRepository } from '../shared/database/repositories/post.repository';

@Module({
  imports: [DatabaseModule, AwsModule],
  providers: [
    SearchService,
    RecommendationService,
    UserRepository,
    VideoRepository,
    PostRepository,
  ],
  controllers: [SearchController],
})
export class SearchModule {}
