-- WEB-AUTH-1: magic-link tokens, sessions, rate counters, account email.
-- Apply after 0001–0003. Do not edit 0001–0003.
--   npx wrangler d1 migrations apply rathor-api-sandbox --local
-- Tokens and session ids are stored as SHA-256 hex only. Raw values never go in D1.

ALTER TABLE accounts ADD COLUMN email TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS accounts_email ON accounts(email) WHERE email IS NOT NULL;

-- Single-use magic-link tokens. consumed_at set atomically on verify.
CREATE TABLE IF NOT EXISTS auth_tokens (
  token_hash TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  consumed_at INTEGER,
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS auth_tokens_email ON auth_tokens(email);
CREATE INDEX IF NOT EXISTS auth_tokens_expires ON auth_tokens(expires_at);

-- Revocable sessions. Lookup is by session_hash (SHA-256 of the cookie value).
CREATE TABLE IF NOT EXISTS sessions (
  session_hash TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  expires_at INTEGER NOT NULL,
  revoked INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS sessions_account ON sessions(account_id);
CREATE INDEX IF NOT EXISTS sessions_expires ON sessions(expires_at);

-- Hour-window counters for POST /v1/auth/request (per email hash and per IP).
CREATE TABLE IF NOT EXISTS rate_counters (
  bucket TEXT NOT NULL,
  window_start INTEGER NOT NULL,
  count INTEGER NOT NULL DEFAULT 0 CHECK (count >= 0),
  PRIMARY KEY (bucket, window_start)
);
CREATE INDEX IF NOT EXISTS rate_counters_window ON rate_counters(window_start);
