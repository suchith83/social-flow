import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { S3Client } from '@aws-sdk/client-s3';
import { MediaConvertClient } from '@aws-sdk/client-mediaconvert';
import { CognitoIdentityProviderClient } from '@aws-sdk/client-cognito-identity-provider';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { ElasticsearchClient } from '@aws-sdk/client-elasticsearch';
import { SQSClient } from '@aws-sdk/client-sqs';
import { SNSClient } from '@aws-sdk/client-sns';
import { CloudFrontClient } from '@aws-sdk/client-cloudfront';

@Injectable()
export class AwsService {
  public readonly s3Client: S3Client;
  public readonly mediaConvertClient: MediaConvertClient;
  public readonly cognitoClient: CognitoIdentityProviderClient;
  public readonly dynamoDBClient: DynamoDBClient;
  public readonly elasticsearchClient: ElasticsearchClient;
  public readonly sqsClient: SQSClient;
  public readonly snsClient: SNSClient;
  public readonly cloudFrontClient: CloudFrontClient;

  constructor(private configService: ConfigService) {
    const awsConfig = this.configService.get('aws');

    // S3 Client
    this.s3Client = new S3Client({
      region: awsConfig.region,
      credentials: {
        accessKeyId: awsConfig.accessKeyId,
        secretAccessKey: awsConfig.secretAccessKey,
      },
    });

    // MediaConvert Client
    this.mediaConvertClient = new MediaConvertClient({
      region: awsConfig.region,
      credentials: {
        accessKeyId: awsConfig.accessKeyId,
        secretAccessKey: awsConfig.secretAccessKey,
      },
      endpoint: awsConfig.mediaConvert.endpoint,
    });

    // Cognito Client
    this.cognitoClient = new CognitoIdentityProviderClient({
      region: awsConfig.cognito.region,
      credentials: {
        accessKeyId: awsConfig.accessKeyId,
        secretAccessKey: awsConfig.secretAccessKey,
      },
    });

    // DynamoDB Client
    this.dynamoDBClient = new DynamoDBClient({
      region: awsConfig.dynamoDB.region,
      credentials: {
        accessKeyId: awsConfig.accessKeyId,
        secretAccessKey: awsConfig.secretAccessKey,
      },
    });

    // Elasticsearch Client
    this.elasticsearchClient = new ElasticsearchClient({
      region: awsConfig.elasticsearch.region,
      credentials: {
        accessKeyId: awsConfig.accessKeyId,
        secretAccessKey: awsConfig.secretAccessKey,
      },
      endpoint: awsConfig.elasticsearch.endpoint,
    });

    // SQS Client
    this.sqsClient = new SQSClient({
      region: awsConfig.sqs.region,
      credentials: {
        accessKeyId: awsConfig.accessKeyId,
        secretAccessKey: awsConfig.secretAccessKey,
      },
    });

    // SNS Client
    this.snsClient = new SNSClient({
      region: awsConfig.sns.region,
      credentials: {
        accessKeyId: awsConfig.accessKeyId,
        secretAccessKey: awsConfig.secretAccessKey,
      },
    });

    // CloudFront Client
    this.cloudFrontClient = new CloudFrontClient({
      region: awsConfig.region,
      credentials: {
        accessKeyId: awsConfig.accessKeyId,
        secretAccessKey: awsConfig.secretAccessKey,
      },
    });
  }

  getRegion(): string {
    return this.configService.get('aws.region');
  }

  getAccountId(): string {
    return this.configService.get('aws.accountId');
  }
}
