-- HOLD-HEADROOM-1: additive. Records usage that cost more than the hold and more than the
-- account's unreserved balance could cover. Do not edit 0001-0007.
CREATE TABLE IF NOT EXISTS settle_shortfalls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  hold_id TEXT NOT NULL,
  actual_credits INTEGER NOT NULL CHECK (actual_credits > 0),
  charged_credits INTEGER NOT NULL CHECK (charged_credits >= 0),
  shortfall_credits INTEGER NOT NULL CHECK (shortfall_credits > 0),
  model TEXT,
  resolved INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS settle_shortfalls_account ON settle_shortfalls(account_id);
CREATE UNIQUE INDEX IF NOT EXISTS settle_shortfalls_hold ON settle_shortfalls(hold_id);
