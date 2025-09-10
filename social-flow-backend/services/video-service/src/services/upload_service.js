class UploadService {
  async initiateSession(videoId) {
    // TODO: Create session in Redis
  }

  async uploadChunk(videoId, chunk) {
    // TODO: Upload to S3 multi-part
  }

  async completeUpload(videoId) {
    // TODO: Complete multi-part, start processing
  }
}

module.exports = UploadService;
