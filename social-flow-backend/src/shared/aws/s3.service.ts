import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AwsService } from './aws.service';
import {
  PutObjectCommand,
  GetObjectCommand,
  DeleteObjectCommand,
  ListObjectsV2Command,
  CopyObjectCommand,
  HeadObjectCommand,
  GetObjectCommandOutput,
} from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { Readable } from 'stream';

@Injectable()
export class S3Service {
  private readonly bucketName: string;

  constructor(
    private awsService: AwsService,
    private configService: ConfigService,
  ) {
    this.bucketName = this.configService.get('aws.s3.bucket');
  }

  async uploadFile(
    key: string,
    file: Buffer | Uint8Array | string | Readable,
    contentType?: string,
    metadata?: Record<string, string>,
  ): Promise<string> {
    const command = new PutObjectCommand({
      Bucket: this.bucketName,
      Key: key,
      Body: file,
      ContentType: contentType,
      Metadata: metadata,
    });

    await this.awsService.s3Client.send(command);
    return `https://${this.bucketName}.s3.${this.awsService.getRegion()}.amazonaws.com/${key}`;
  }

  async uploadFileWithSignedUrl(
    key: string,
    file: Buffer | Uint8Array | string | Readable,
    contentType?: string,
    metadata?: Record<string, string>,
  ): Promise<{ url: string; signedUrl: string }> {
    const url = await this.uploadFile(key, file, contentType, metadata);
    const signedUrl = await this.getSignedUrl(key, 3600); // 1 hour
    return { url, signedUrl };
  }

  async getFile(key: string): Promise<GetObjectCommandOutput> {
    const command = new GetObjectCommand({
      Bucket: this.bucketName,
      Key: key,
    });

    return this.awsService.s3Client.send(command);
  }

  async downloadFile(key: string): Promise<Buffer> {
    const response = await this.getFile(key);
    const chunks: Uint8Array[] = [];
    
    if (response.Body) {
      for await (const chunk of response.Body as any) {
        chunks.push(chunk);
      }
    }

    return Buffer.concat(chunks);
  }

  async deleteFile(key: string): Promise<void> {
    const command = new DeleteObjectCommand({
      Bucket: this.bucketName,
      Key: key,
    });

    await this.awsService.s3Client.send(command);
  }

  async deleteFiles(keys: string[]): Promise<void> {
    const promises = keys.map(key => this.deleteFile(key));
    await Promise.all(promises);
  }

  async listFiles(prefix?: string, maxKeys?: number): Promise<string[]> {
    const command = new ListObjectsV2Command({
      Bucket: this.bucketName,
      Prefix: prefix,
      MaxKeys: maxKeys,
    });

    const response = await this.awsService.s3Client.send(command);
    return response.Contents?.map(obj => obj.Key!) || [];
  }

  async copyFile(sourceKey: string, destinationKey: string): Promise<string> {
    const command = new CopyObjectCommand({
      Bucket: this.bucketName,
      CopySource: `${this.bucketName}/${sourceKey}`,
      Key: destinationKey,
    });

    await this.awsService.s3Client.send(command);
    return `https://${this.bucketName}.s3.${this.awsService.getRegion()}.amazonaws.com/${destinationKey}`;
  }

  async fileExists(key: string): Promise<boolean> {
    try {
      const command = new HeadObjectCommand({
        Bucket: this.bucketName,
        Key: key,
      });

      await this.awsService.s3Client.send(command);
      return true;
    } catch (error) {
      return false;
    }
  }

  async getFileMetadata(key: string): Promise<Record<string, any>> {
    const command = new HeadObjectCommand({
      Bucket: this.bucketName,
      Key: key,
    });

    const response = await this.awsService.s3Client.send(command);
    return {
      contentType: response.ContentType,
      contentLength: response.ContentLength,
      lastModified: response.LastModified,
      metadata: response.Metadata,
      etag: response.ETag,
    };
  }

  async getSignedUrl(key: string, expiresIn: number = 3600): Promise<string> {
    const command = new GetObjectCommand({
      Bucket: this.bucketName,
      Key: key,
    });

    return getSignedUrl(this.awsService.s3Client, command, { expiresIn });
  }

  async getSignedUploadUrl(
    key: string,
    contentType: string,
    expiresIn: number = 3600,
  ): Promise<string> {
    const command = new PutObjectCommand({
      Bucket: this.bucketName,
      Key: key,
      ContentType: contentType,
    });

    return getSignedUrl(this.awsService.s3Client, command, { expiresIn });
  }

  // Video-specific methods
  async uploadVideo(
    videoId: string,
    file: Buffer | Uint8Array | string | Readable,
    originalName: string,
  ): Promise<string> {
    const key = `videos/${videoId}/${originalName}`;
    return this.uploadFile(key, file, 'video/mp4');
  }

  async uploadVideoThumbnail(
    videoId: string,
    thumbnail: Buffer | Uint8Array | string | Readable,
    size: string = 'default',
  ): Promise<string> {
    const key = `videos/${videoId}/thumbnails/${size}.jpg`;
    return this.uploadFile(key, thumbnail, 'image/jpeg');
  }

  async uploadUserAvatar(
    userId: string,
    avatar: Buffer | Uint8Array | string | Readable,
    originalName: string,
  ): Promise<string> {
    const key = `avatars/${userId}/${originalName}`;
    return this.uploadFile(key, avatar, 'image/jpeg');
  }

  async uploadPostMedia(
    postId: string,
    media: Buffer | Uint8Array | string | Readable,
    originalName: string,
    mediaType: 'image' | 'video' = 'image',
  ): Promise<string> {
    const key = `posts/${postId}/${originalName}`;
    const contentType = mediaType === 'video' ? 'video/mp4' : 'image/jpeg';
    return this.uploadFile(key, media, contentType);
  }

  // Generate CDN URLs
  getCdnUrl(key: string): string {
    const cloudFrontDomain = this.configService.get('aws.cloudFront.domain');
    if (cloudFrontDomain) {
      return `https://${cloudFrontDomain}/${key}`;
    }
    return `https://${this.bucketName}.s3.${this.awsService.getRegion()}.amazonaws.com/${key}`;
  }

  getVideoUrl(videoId: string, filename: string): string {
    return this.getCdnUrl(`videos/${videoId}/${filename}`);
  }

  getThumbnailUrl(videoId: string, size: string = 'default'): string {
    return this.getCdnUrl(`videos/${videoId}/thumbnails/${size}.jpg`);
  }

  getAvatarUrl(userId: string, filename: string): string {
    return this.getCdnUrl(`avatars/${userId}/${filename}`);
  }

  getPostMediaUrl(postId: string, filename: string): string {
    return this.getCdnUrl(`posts/${postId}/${filename}`);
  }
}
