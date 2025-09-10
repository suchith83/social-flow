const AWS = require('aws-sdk');
const s3 = new AWS.S3();

async function uploadToS3(file, bucket, key) {
  await s3.putObject({
    Bucket: bucket,
    Key: key,
    Body: file
  }).promise();
}

module.exports = { uploadToS3 };
