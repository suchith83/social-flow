import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { BullModule } from '@nestjs/bull';
import { ThrottlerModule } from '@nestjs/throttler';
import { ScheduleModule } from '@nestjs/schedule';

// Shared modules
import { DatabaseModule } from './shared/database/database.module';
import { RedisModule } from './shared/redis/redis.module';
import { AwsModule } from './shared/aws/aws.module';
import { LoggerModule } from './shared/logger/logger.module';

// Feature modules
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { VideosModule } from './videos/videos.module';
import { PostsModule } from './posts/posts.module';
import { AdsModule } from './ads/ads.module';
import { PaymentsModule } from './payments/payments.module';
import { NotificationsModule } from './notifications/notifications.module';
import { AnalyticsModule } from './analytics/analytics.module';
import { AdminModule } from './admin/admin.module';
import { SearchModule } from './search/search.module';
import { RealtimeModule } from './realtime/realtime.module';
import { ModerationModule } from './moderation/moderation.module';

// Configuration
import { databaseConfig } from './shared/config/database.config';
import { redisConfig } from './shared/config/redis.config';
import { awsConfig } from './shared/config/aws.config';
import { appConfig } from './shared/config/app.config';

@Module({
  imports: [
    // Configuration
    ConfigModule.forRoot({
      isGlobal: true,
      load: [databaseConfig, redisConfig, awsConfig, appConfig],
      envFilePath: ['.env.local', '.env'],
    }),

    // Database
    TypeOrmModule.forRootAsync({
      useFactory: (configService) => ({
        type: 'postgres',
        host: configService.get('database.host'),
        port: configService.get('database.port'),
        username: configService.get('database.username'),
        password: configService.get('database.password'),
        database: configService.get('database.name'),
        entities: [__dirname + '/**/*.entity{.ts,.js}'],
        synchronize: configService.get('app.nodeEnv') === 'development',
        logging: configService.get('app.nodeEnv') === 'development',
        ssl: configService.get('database.ssl'),
      }),
      inject: ['CONFIGURATION(database)'],
    }),

    // Redis for caching and queues
    BullModule.forRootAsync({
      useFactory: (configService) => ({
        redis: {
          host: configService.get('redis.host'),
          port: configService.get('redis.port'),
          password: configService.get('redis.password'),
        },
      }),
      inject: ['CONFIGURATION(redis)'],
    }),

    // Rate limiting
    ThrottlerModule.forRootAsync({
      useFactory: (configService) => ({
        ttl: configService.get('app.rateLimitTtl'),
        limit: configService.get('app.rateLimitCount'),
      }),
      inject: ['CONFIGURATION(app)'],
    }),

    // Task scheduling
    ScheduleModule.forRoot(),

    // Shared modules
    DatabaseModule,
    RedisModule,
    AwsModule,
    LoggerModule,

    // Feature modules
    AuthModule,
    UsersModule,
    VideosModule,
    PostsModule,
    AdsModule,
    PaymentsModule,
    NotificationsModule,
    AnalyticsModule,
    AdminModule,
    SearchModule,
    RealtimeModule,
    ModerationModule,
  ],
})
export class AppModule {}
