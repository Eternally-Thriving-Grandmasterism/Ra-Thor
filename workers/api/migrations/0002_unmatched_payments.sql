-- PRICING-2: Stripe payments the webhook refused to grant credits for (unknown price id,
-- amount/currency mismatch, wrong mode, ...). Recorded for manual review; Stripe gets a 200.
CREATE TABLE IF NOT EXISTS unmatched_payments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  stripe_event_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  reason TEXT NOT NULL,
  account_id TEXT,
  price_id TEXT,
  amount INTEGER,                    -- smallest currency unit, as reported by Stripe
  currency TEXT,
  mode TEXT,                         -- checkout mode or invoice billing_reason
  metadata_credits TEXT,             -- client-side metadata, recorded but never trusted
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS unmatched_event ON unmatched_payments(stripe_event_id);
