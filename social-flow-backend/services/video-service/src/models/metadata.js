const mongoose = require('mongoose');

const metadataSchema = new mongoose.Schema({
  videoId: String,
  duration: Number,
  resolution: String,
  format: String
});

module.exports = mongoose.model('Metadata', metadataSchema);
