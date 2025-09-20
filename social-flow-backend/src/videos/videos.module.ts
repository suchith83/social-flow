import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bull';
import { VideosController } from './videos.controller';
import { VideosService } from './videos.service';
import { VideoProcessingService } from './video-processing.service';
import { VideoProcessingProcessor } from './processors/video-processing.processor';
import { ThumbnailProcessor } from './processors/thumbnail.processor';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Module({
  imports: [
    BullModule.registerQueue(
      { name: 'video-processing' },
      { name: 'thumbnail-generation' },
    ),
  ],
  controllers: [VideosController],
  providers: [
    VideosService,
    VideoProcessingService,
    VideoProcessingProcessor,
    ThumbnailProcessor,
    JwtAuthGuard,
  ],
  exports: [VideosService, VideoProcessingService],
})
export class VideosModule {}
