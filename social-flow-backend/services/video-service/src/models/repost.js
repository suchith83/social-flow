const mongoose = require('mongoose');

const repostSchema = new mongoose.Schema({
  videoId: String,
  userId: String,
  timestamp: Date
});

module.exports = mongoose.model('Repost', repostSchema);
