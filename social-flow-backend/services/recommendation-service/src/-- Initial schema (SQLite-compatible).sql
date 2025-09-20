-- Initial schema (SQLite-compatible). For production use proper migration tooling (Alembic/Flyway).

PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- Users table
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  display_name TEXT,
  salt TEXT,
  pwd_hash TEXT,
  role TEXT DEFAULT 'user',
  created_at REAL NOT NULL
);

-- Videos table
CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  uploader_id TEXT,
  duration REAL,
  s3_key TEXT,
  uploaded_at REAL NOT NULL,
  status TEXT DEFAULT 'processing',
  FOREIGN KEY(uploader_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Posts (micro-posts / tweets)
CREATE TABLE IF NOT EXISTS posts (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  text TEXT,
  media_ref TEXT,
  created_at REAL NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Recommendation feedback (used by recommendation worker)
CREATE TABLE IF NOT EXISTS recommendation_feedback (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  item_id TEXT,
  action TEXT,
  timestamp INTEGER,
  payload_json TEXT,
  created_at REAL NOT NULL
);

-- Payments / transactions (lightweight)
CREATE TABLE IF NOT EXISTS payments (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  amount_cents INTEGER,
  currency TEXT,
  provider TEXT,
  provider_charge_id TEXT,
  status TEXT,
  created_at REAL NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Subscriptions / creator payouts
CREATE TABLE IF NOT EXISTS subscriptions (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  tier TEXT,
  active INTEGER DEFAULT 1,
  started_at REAL,
  ended_at REAL
);

-- Lightweight analytics events store (development)
CREATE TABLE IF NOT EXISTS analytics_events (
  id TEXT PRIMARY KEY,
  name TEXT,
  value REAL,
  tags_json TEXT,
  timestamp REAL,
  created_at REAL NOT NULL
);

COMMIT;
