const mongoose = require('mongoose');

const videoSchema = new mongoose.Schema({
  title: String,
  description: String,
  userId: String,
  views: { type: Number, default: 0 },
  likes: { type: Number, default: 0 },
  shares: { type: Number, default: 0 },
  adsEnabled: Boolean,
  paymentRequired: Boolean
});

module.exports = mongoose.model('Video', videoSchema);
