import { Module } from '@nestjs/common';
import { AwsService } from './aws.service';
import { S3Service } from './s3.service';
import { MediaConvertService } from './media-convert.service';
import { CognitoService } from './cognito.service';
import { DynamoDBService } from './dynamodb.service';
import { ElasticsearchService } from './elasticsearch.service';
import { SQSService } from './sqs.service';
import { SNSService } from './sns.service';
import { CloudFrontService } from './cloudfront.service';

@Module({
  providers: [
    AwsService,
    S3Service,
    MediaConvertService,
    CognitoService,
    DynamoDBService,
    ElasticsearchService,
    SQSService,
    SNSService,
    CloudFrontService,
  ],
  exports: [
    AwsService,
    S3Service,
    MediaConvertService,
    CognitoService,
    DynamoDBService,
    ElasticsearchService,
    SQSService,
    SNSService,
    CloudFrontService,
  ],
})
export class AwsModule {}
