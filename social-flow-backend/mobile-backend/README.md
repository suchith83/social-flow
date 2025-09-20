# Mobile Backend (Lightweight)

Purpose
- Small backend optimized for mobile clients (Flutter). Provides:
  - Device registration
  - Lightweight sync endpoint
  - Push notification sending (FCM optional)
  - Notification retrieval

Quick start
1. Create venv and install dependencies:
   python -m venv .venv
   . .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
   pip install fastapi uvicorn

2. Run:
   uvicorn mobile_backend.main:app --host 0.0.0.0 --port 8101 --reload

Configuration
- SF_PUSH_PROVIDER: 'fcm' or 'noop' (default 'fcm')
- SF_PUSH_CREDENTIALS: path to Firebase service account JSON (optional)

Notes
- This is a development-friendly implementation using in-memory stores. For production replace stores with a persistent DB, use a proper push provider config, and add authentication & rate limiting.
