/**
 * Optional helper to create objects in storage backends (S3/Azure/GCS) before tests run.
 * This is a best-effort module and should be wired into global setup if desired.
 *
 * Example usage: import and call createTestObjects() within global-setup.
 *
 * NOTE: This file uses environment variables for credentials:
 *   - AWS_* for S3
 *   - AZURE_* for Azure Blob
 *   - GCP_* for GCS
 */

import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import AWS from 'aws-sdk'; // ensure aws-sdk installed if used
// For brevity, only S3 example implemented. Extend as needed.

dotenv.config();

export async function createS3TestObject(key: string, content: Buffer | string) {
  const bucket = process.env.AWS_S3_BUCKET!;
  if (!bucket) {
    console.log('AWS_S3_BUCKET not set; skipping S3 fixtures.');
    return;
  }
  const s3 = new AWS.S3({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION || 'us-east-1'
  });
  await s3
    .putObject({
      Bucket: bucket,
      Key: key,
      Body: content
    })
    .promise();
}
