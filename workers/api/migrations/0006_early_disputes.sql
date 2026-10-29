-- SPROOF-FIX-DISPUTE-ORDER: durable early-dispute match.
-- Apply after 0001–0005. Do not edit 0001–0005.
--   npx wrangler d1 migrations apply rathor-api-sandbox --local
-- A charge.dispute.* event that arrives before its grant (invoice.paid /
-- checkout completion) has no credit_grants row to match. It is stored here,
-- keyed by payment_intent_id and charge_id, one row per dispute (idempotent on
-- dispute_id; a replayed event only refreshes the row). grant() reads this table
-- in the same D1 batch that writes credit_grants and applies the freeze (open),
-- nothing (won / warning_closed) or the lost claw + freeze (lost / charge_refunded).
-- Terminal statuses are sticky, like stripe_disputes.
CREATE TABLE IF NOT EXISTS early_disputes (
  dispute_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  payment_intent_id TEXT,
  charge_id TEXT,
  amount INTEGER,
  currency TEXT,
  stripe_event_id TEXT NOT NULL,
  applied_event_id TEXT,
  applied_account_id TEXT,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS early_disputes_pi ON early_disputes(payment_intent_id);
CREATE INDEX IF NOT EXISTS early_disputes_ch ON early_disputes(charge_id);
