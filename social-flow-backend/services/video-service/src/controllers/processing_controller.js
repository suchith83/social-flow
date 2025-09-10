const express = require('express');

// Objective: Handle video processing status

// Input Cases:
// - GET /api/v1/videos/:id/processing - Get processing status

// Output Cases:
// - Success: Processing status, progress
// - Error: Not found

async function getProcessingStatus(req, res) {
    // TODO: Check MediaConvert job status
}

module.exports = {
    getProcessingStatus
};
