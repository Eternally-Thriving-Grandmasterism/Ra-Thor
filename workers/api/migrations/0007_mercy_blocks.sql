-- MERCY-LAUNCH-1: Mercy-Checks block audit. Apply after 0001-0006 and BEFORE deploying
-- the Worker code that writes it. Do not edit 0001-0006.
--   npx wrangler d1 migrations apply rathor-api-sandbox --local
-- One row per block. NO content and NO matched text are ever stored here.
-- request_id is server-generated (crypto.randomUUID), never derived from content.
-- account_id is NULL for request-stage blocks (no account is created for them).
-- Retention: delete rows older than 90 days (see workers/api/README.md).
CREATE TABLE IF NOT EXISTS mercy_blocks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  stage TEXT NOT NULL CHECK (stage IN ('request', 'response')),
  rule_id TEXT NOT NULL,
  rules_version TEXT NOT NULL,
  model TEXT,
  account_id TEXT,
  charged_credits INTEGER NOT NULL DEFAULT 0,
  request_id TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS mercy_blocks_ts ON mercy_blocks(ts);
