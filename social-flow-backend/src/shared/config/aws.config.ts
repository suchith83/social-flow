import { registerAs } from '@nestjs/config';

export const awsConfig = registerAs('aws', () => ({
  region: process.env.AWS_REGION || 'us-west-2',
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  s3: {
    bucket: process.env.AWS_S3_BUCKET || 'social-flow-media',
    region: process.env.AWS_S3_REGION || process.env.AWS_REGION || 'us-west-2',
  },
  cloudFront: {
    distributionId: process.env.AWS_CLOUDFRONT_DISTRIBUTION_ID,
    domain: process.env.AWS_CLOUDFRONT_DOMAIN,
  },
  mediaConvert: {
    role: process.env.AWS_MEDIACONVERT_ROLE,
    endpoint: process.env.AWS_MEDIACONVERT_ENDPOINT,
  },
  cognito: {
    userPoolId: process.env.AWS_COGNITO_USER_POOL_ID,
    clientId: process.env.AWS_COGNITO_CLIENT_ID,
    region: process.env.AWS_COGNITO_REGION || process.env.AWS_REGION || 'us-west-2',
  },
  dynamoDB: {
    region: process.env.AWS_DYNAMODB_REGION || process.env.AWS_REGION || 'us-west-2',
  },
  elasticsearch: {
    endpoint: process.env.AWS_ELASTICSEARCH_ENDPOINT,
    region: process.env.AWS_ELASTICSEARCH_REGION || process.env.AWS_REGION || 'us-west-2',
  },
  sqs: {
    region: process.env.AWS_SQS_REGION || process.env.AWS_REGION || 'us-west-2',
  },
  sns: {
    region: process.env.AWS_SNS_REGION || process.env.AWS_REGION || 'us-west-2',
  },
}));
