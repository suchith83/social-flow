import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AwsService } from './aws.service';
import {
  PublishCommand,
  CreateTopicCommand,
  DeleteTopicCommand,
  SubscribeCommand,
  UnsubscribeCommand,
  ListSubscriptionsByTopicCommand,
} from '@aws-sdk/client-sns';

@Injectable()
export class SNSService {
  constructor(
    private awsService: AwsService,
    private configService: ConfigService,
  ) {}

  async publishMessage(topicArn: string, message: string, subject?: string): Promise<any> {
    const command = new PublishCommand({
      TopicArn: topicArn,
      Message: message,
      Subject: subject,
    });

    return this.awsService.snsClient.send(command);
  }

  async createTopic(name: string): Promise<string> {
    const command = new CreateTopicCommand({
      Name: name,
    });

    const response = await this.awsService.snsClient.send(command);
    return response.TopicArn || '';
  }

  async deleteTopic(topicArn: string): Promise<any> {
    const command = new DeleteTopicCommand({
      TopicArn: topicArn,
    });

    return this.awsService.snsClient.send(command);
  }

  async subscribe(topicArn: string, protocol: string, endpoint: string): Promise<string> {
    const command = new SubscribeCommand({
      TopicArn: topicArn,
      Protocol: protocol,
      Endpoint: endpoint,
    });

    const response = await this.awsService.snsClient.send(command);
    return response.SubscriptionArn || '';
  }

  async unsubscribe(subscriptionArn: string): Promise<any> {
    const command = new UnsubscribeCommand({
      SubscriptionArn: subscriptionArn,
    });

    return this.awsService.snsClient.send(command);
  }

  async listSubscriptions(topicArn: string): Promise<any[]> {
    const command = new ListSubscriptionsByTopicCommand({
      TopicArn: topicArn,
    });

    const response = await this.awsService.snsClient.send(command);
    return response.Subscriptions || [];
  }
}
