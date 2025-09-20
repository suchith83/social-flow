import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AwsService } from './aws.service';
import { Client } from '@elastic/elasticsearch';

@Injectable()
export class ElasticsearchService {
  private readonly client: Client;

  constructor(
    private awsService: AwsService,
    private configService: ConfigService,
  ) {
    const endpoint = this.configService.get('aws.elasticsearch.endpoint');
    this.client = new Client({
      node: endpoint,
      auth: {
        username: this.configService.get('aws.accessKeyId'),
        password: this.configService.get('aws.secretAccessKey'),
      },
    });
  }

  async index(indexName: string, id: string, document: Record<string, any>): Promise<any> {
    return this.client.index({
      index: indexName,
      id,
      body: document,
    });
  }

  async get(indexName: string, id: string): Promise<any> {
    return this.client.get({
      index: indexName,
      id,
    });
  }

  async search(indexName: string, query: Record<string, any>): Promise<any> {
    return this.client.search({
      index: indexName,
      body: query,
    });
  }

  async update(indexName: string, id: string, document: Record<string, any>): Promise<any> {
    return this.client.update({
      index: indexName,
      id,
      body: {
        doc: document,
      },
    });
  }

  async delete(indexName: string, id: string): Promise<any> {
    return this.client.delete({
      index: indexName,
      id,
    });
  }

  async bulk(operations: any[]): Promise<any> {
    return this.client.bulk({
      body: operations,
    });
  }

  async createIndex(indexName: string, mapping: Record<string, any>): Promise<any> {
    return this.client.indices.create({
      index: indexName,
      body: mapping,
    });
  }

  async deleteIndex(indexName: string): Promise<any> {
    return this.client.indices.delete({
      index: indexName,
    });
  }

  async indexExists(indexName: string): Promise<boolean> {
    return this.client.indices.exists({
      index: indexName,
    });
  }
}
