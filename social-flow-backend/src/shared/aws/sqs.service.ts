import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AwsService } from './aws.service';
import {
  SendMessageCommand,
  ReceiveMessageCommand,
  DeleteMessageCommand,
  GetQueueAttributesCommand,
  CreateQueueCommand,
  DeleteQueueCommand,
} from '@aws-sdk/client-sqs';

@Injectable()
export class SQSService {
  constructor(
    private awsService: AwsService,
    private configService: ConfigService,
  ) {}

  async sendMessage(queueUrl: string, messageBody: string, messageAttributes?: Record<string, any>): Promise<any> {
    const command = new SendMessageCommand({
      QueueUrl: queueUrl,
      MessageBody: messageBody,
      MessageAttributes: messageAttributes,
    });

    return this.awsService.sqsClient.send(command);
  }

  async receiveMessages(queueUrl: string, maxMessages: number = 10, waitTimeSeconds: number = 0): Promise<any[]> {
    const command = new ReceiveMessageCommand({
      QueueUrl: queueUrl,
      MaxNumberOfMessages: maxMessages,
      WaitTimeSeconds: waitTimeSeconds,
    });

    const response = await this.awsService.sqsClient.send(command);
    return response.Messages || [];
  }

  async deleteMessage(queueUrl: string, receiptHandle: string): Promise<any> {
    const command = new DeleteMessageCommand({
      QueueUrl: queueUrl,
      ReceiptHandle: receiptHandle,
    });

    return this.awsService.sqsClient.send(command);
  }

  async getQueueAttributes(queueUrl: string, attributeNames: string[] = ['All']): Promise<any> {
    const command = new GetQueueAttributesCommand({
      QueueUrl: queueUrl,
      AttributeNames: attributeNames,
    });

    return this.awsService.sqsClient.send(command);
  }

  async createQueue(queueName: string, attributes?: Record<string, string>): Promise<string> {
    const command = new CreateQueueCommand({
      QueueName: queueName,
      Attributes: attributes,
    });

    const response = await this.awsService.sqsClient.send(command);
    return response.QueueUrl || '';
  }

  async deleteQueue(queueUrl: string): Promise<any> {
    const command = new DeleteQueueCommand({
      QueueUrl: queueUrl,
    });

    return this.awsService.sqsClient.send(command);
  }
}
