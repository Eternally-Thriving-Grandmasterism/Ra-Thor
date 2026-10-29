-- Ra-Thor API Worker — D1 schema (SANDBOX). Apply with:
--   npx wrangler d1 migrations apply rathor-api-sandbox --local
-- Balances are integers in credits (1 credit = $0.001). available = balance - SUM(open holds).
CREATE TABLE IF NOT EXISTS accounts (
  id TEXT PRIMARY KEY,
  balance INTEGER NOT NULL DEFAULT 0 CHECK (balance >= 0),
  stripe_customer_id TEXT UNIQUE,
  subscription_id TEXT,
  subscription_status TEXT,          -- active | trialing | past_due | canceled | ...
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  delta INTEGER NOT NULL,            -- +grant / -charge
  reason TEXT NOT NULL,              -- trial | checkout | renewal | chat
  hold_id TEXT,
  stripe_event_id TEXT,
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS ledger_account ON ledger(account_id);

CREATE TABLE IF NOT EXISTS holds (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  amount INTEGER NOT NULL CHECK (amount > 0),
  status TEXT NOT NULL DEFAULT 'open', -- open | settled | released
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS holds_open ON holds(account_id, status);

-- Idempotency: each Stripe event id is processed at most once.
CREATE TABLE IF NOT EXISTS stripe_events (
  event_id TEXT NOT NULL UNIQUE,
  type TEXT NOT NULL,
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

-- Per-customer API keys. Only the SHA-256 hex of the key is stored.
CREATE TABLE IF NOT EXISTS api_keys (
  key_hash TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  revoked INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
