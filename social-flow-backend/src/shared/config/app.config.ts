import { registerAs } from '@nestjs/config';

export const appConfig = registerAs('app', () => ({
  nodeEnv: process.env.NODE_ENV || 'development',
  port: parseInt(process.env.PORT, 10) || 3000,
  jwtSecret: process.env.JWT_SECRET || 'your-super-secret-jwt-key',
  jwtExpiresIn: process.env.JWT_EXPIRES_IN || '7d',
  refreshTokenExpiresIn: process.env.REFRESH_TOKEN_EXPIRES_IN || '30d',
  rateLimitTtl: parseInt(process.env.RATE_LIMIT_TTL, 10) || 60,
  rateLimitCount: parseInt(process.env.RATE_LIMIT_COUNT, 10) || 100,
  allowedOrigins: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  maxFileSize: parseInt(process.env.MAX_FILE_SIZE, 10) || 100 * 1024 * 1024, // 100MB
  maxVideoSize: parseInt(process.env.MAX_VIDEO_SIZE, 10) || 500 * 1024 * 1024, // 500MB
  videoFormats: process.env.VIDEO_FORMATS?.split(',') || ['mp4', 'mov', 'avi', 'mkv'],
  imageFormats: process.env.IMAGE_FORMATS?.split(',') || ['jpg', 'jpeg', 'png', 'gif', 'webp'],
  stripeSecretKey: process.env.STRIPE_SECRET_KEY,
  stripeWebhookSecret: process.env.STRIPE_WEBHOOK_SECRET,
  emailFrom: process.env.EMAIL_FROM || 'noreply@socialflow.com',
  emailHost: process.env.EMAIL_HOST,
  emailPort: parseInt(process.env.EMAIL_PORT, 10) || 587,
  emailUser: process.env.EMAIL_USER,
  emailPassword: process.env.EMAIL_PASSWORD,
  fcmServerKey: process.env.FCM_SERVER_KEY,
  apnsKeyId: process.env.APNS_KEY_ID,
  apnsTeamId: process.env.APNS_TEAM_ID,
  apnsBundleId: process.env.APNS_BUNDLE_ID,
  apnsPrivateKey: process.env.APNS_PRIVATE_KEY,
}));
