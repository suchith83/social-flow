const ffmpeg = require('fluent-ffmpeg');

function extractThumbnail(videoPath, outputPath) {
  ffmpeg(videoPath)
    .screenshots({
      count: 1,
      folder: outputPath
    });
}

module.exports = { extractThumbnail };
