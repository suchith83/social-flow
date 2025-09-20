import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AwsService } from './aws.service';
import {
  PutItemCommand,
  GetItemCommand,
  UpdateItemCommand,
  DeleteItemCommand,
  QueryCommand,
  ScanCommand,
} from '@aws-sdk/client-dynamodb';
import { marshall, unmarshall } from '@aws-sdk/util-dynamodb';

@Injectable()
export class DynamoDBService {
  constructor(
    private awsService: AwsService,
    private configService: ConfigService,
  ) {}

  async putItem(tableName: string, item: Record<string, any>): Promise<any> {
    const command = new PutItemCommand({
      TableName: tableName,
      Item: marshall(item),
    });

    return this.awsService.dynamoDBClient.send(command);
  }

  async getItem(tableName: string, key: Record<string, any>): Promise<any> {
    const command = new GetItemCommand({
      TableName: tableName,
      Key: marshall(key),
    });

    const response = await this.awsService.dynamoDBClient.send(command);
    return response.Item ? unmarshall(response.Item) : null;
  }

  async updateItem(
    tableName: string,
    key: Record<string, any>,
    updateExpression: string,
    expressionAttributeValues?: Record<string, any>,
    expressionAttributeNames?: Record<string, string>,
  ): Promise<any> {
    const command = new UpdateItemCommand({
      TableName: tableName,
      Key: marshall(key),
      UpdateExpression: updateExpression,
      ExpressionAttributeValues: expressionAttributeValues ? marshall(expressionAttributeValues) : undefined,
      ExpressionAttributeNames: expressionAttributeNames,
    });

    return this.awsService.dynamoDBClient.send(command);
  }

  async deleteItem(tableName: string, key: Record<string, any>): Promise<any> {
    const command = new DeleteItemCommand({
      TableName: tableName,
      Key: marshall(key),
    });

    return this.awsService.dynamoDBClient.send(command);
  }

  async query(
    tableName: string,
    keyConditionExpression: string,
    expressionAttributeValues?: Record<string, any>,
    expressionAttributeNames?: Record<string, string>,
    indexName?: string,
  ): Promise<any[]> {
    const command = new QueryCommand({
      TableName: tableName,
      KeyConditionExpression: keyConditionExpression,
      ExpressionAttributeValues: expressionAttributeValues ? marshall(expressionAttributeValues) : undefined,
      ExpressionAttributeNames: expressionAttributeNames,
      IndexName: indexName,
    });

    const response = await this.awsService.dynamoDBClient.send(command);
    return response.Items?.map(item => unmarshall(item)) || [];
  }

  async scan(
    tableName: string,
    filterExpression?: string,
    expressionAttributeValues?: Record<string, any>,
    expressionAttributeNames?: Record<string, string>,
  ): Promise<any[]> {
    const command = new ScanCommand({
      TableName: tableName,
      FilterExpression: filterExpression,
      ExpressionAttributeValues: expressionAttributeValues ? marshall(expressionAttributeValues) : undefined,
      ExpressionAttributeNames: expressionAttributeNames,
    });

    const response = await this.awsService.dynamoDBClient.send(command);
    return response.Items?.map(item => unmarshall(item)) || [];
  }
}
