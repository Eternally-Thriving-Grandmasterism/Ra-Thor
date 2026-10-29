-- PRICING-3: the paid $1 trial (STRIPE_PRICE_TRIAL) is granted at most once per account.
-- The PRIMARY KEY makes a second redemption fail inside the grant transaction (atomic, no race);
-- the webhook then records it in unmatched_payments (reason trial_already_used) and answers 200.
CREATE TABLE IF NOT EXISTS trial_redemptions (
  account_id TEXT PRIMARY KEY REFERENCES accounts(id),
  stripe_event_id TEXT NOT NULL,
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
