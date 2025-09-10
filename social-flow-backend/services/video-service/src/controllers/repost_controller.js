const express = require('express');

// Objective: Handle video reposts

// Input Cases:
// - POST /api/v1/videos/:id/repost - Repost video

// Output Cases:
// - Success: Repost ID
// - Error: Permission denied

async function repostVideo(req, res) {
    // TODO: Create repost record, publish event
}

module.exports = {
    repostVideo
};
