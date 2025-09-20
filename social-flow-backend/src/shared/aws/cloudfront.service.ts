import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AwsService } from './aws.service';
import {
  CreateInvalidationCommand,
  GetInvalidationCommand,
  ListInvalidationsCommand,
  GetDistributionCommand,
  ListDistributionsCommand,
} from '@aws-sdk/client-cloudfront';

@Injectable()
export class CloudFrontService {
  private readonly distributionId: string;

  constructor(
    private awsService: AwsService,
    private configService: ConfigService,
  ) {
    this.distributionId = this.configService.get('aws.cloudFront.distributionId');
  }

  async createInvalidation(paths: string[]): Promise<string> {
    const command = new CreateInvalidationCommand({
      DistributionId: this.distributionId,
      InvalidationBatch: {
        Paths: {
          Quantity: paths.length,
          Items: paths,
        },
        CallerReference: Date.now().toString(),
      },
    });

    const response = await this.awsService.cloudFrontClient.send(command);
    return response.Invalidation?.Id || '';
  }

  async getInvalidation(invalidationId: string): Promise<any> {
    const command = new GetInvalidationCommand({
      DistributionId: this.distributionId,
      Id: invalidationId,
    });

    return this.awsService.cloudFrontClient.send(command);
  }

  async listInvalidations(maxItems?: number): Promise<any[]> {
    const command = new ListInvalidationsCommand({
      DistributionId: this.distributionId,
      MaxItems: maxItems,
    });

    const response = await this.awsService.cloudFrontClient.send(command);
    return response.InvalidationList?.Items || [];
  }

  async getDistribution(): Promise<any> {
    const command = new GetDistributionCommand({
      Id: this.distributionId,
    });

    return this.awsService.cloudFrontClient.send(command);
  }

  async listDistributions(): Promise<any[]> {
    const command = new ListDistributionsCommand({});

    const response = await this.awsService.cloudFrontClient.send(command);
    return response.DistributionList?.Items || [];
  }
}
