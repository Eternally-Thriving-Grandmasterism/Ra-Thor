-- REFUND-1: refund clawback, dispute freeze, unmatched.detail, unique unmatched index.
-- Apply after 0001–0004. Do not edit 0001–0004.
--   npx wrangler d1 migrations apply rathor-api-sandbox --local
-- Mechanical ledger only. Refund policy copy is TERMS-1, not this migration.
--
-- 0001–0005 are meant for a fresh D1. If this file is applied to an existing
-- unmatched_payments table that already has duplicate (stripe_event_id, reason)
-- rows, dedupe before creating the unique index:
--   DELETE FROM unmatched_payments
--   WHERE stripe_event_id IS NOT NULL
--     AND id NOT IN (
--       SELECT MIN(id) FROM unmatched_payments
--       WHERE stripe_event_id IS NOT NULL
--       GROUP BY stripe_event_id, reason
--     );

ALTER TABLE accounts ADD COLUMN frozen INTEGER NOT NULL DEFAULT 0;
ALTER TABLE accounts ADD COLUMN freeze_reason TEXT;

ALTER TABLE unmatched_payments ADD COLUMN detail TEXT;

-- Idempotent unmatched rows per (event, reason). Inserts use INSERT OR IGNORE
-- or WHERE NOT EXISTS so a replay does not fail the batch. The DELETE above
-- (comment) is the existing-DB path; a fresh D1 has no rows to collide.
DELETE FROM unmatched_payments
 WHERE stripe_event_id IS NOT NULL
   AND id NOT IN (
     SELECT MIN(id) FROM unmatched_payments
     WHERE stripe_event_id IS NOT NULL
     GROUP BY stripe_event_id, reason
   );

CREATE UNIQUE INDEX IF NOT EXISTS unmatched_event_reason
  ON unmatched_payments(stripe_event_id, reason)
  WHERE stripe_event_id IS NOT NULL;

-- One row per credit grant, linked to the Stripe objects the webhook saw.
-- clawed_credits is the sum actually removed (never more than the live balance).
CREATE TABLE IF NOT EXISTS credit_grants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  stripe_event_id TEXT NOT NULL,
  payment_intent_id TEXT,
  charge_id TEXT,
  checkout_session_id TEXT,
  granted_credits INTEGER NOT NULL,
  amount_paid INTEGER NOT NULL,
  clawed_credits INTEGER NOT NULL DEFAULT 0 CHECK (clawed_credits >= 0),
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS credit_grants_pi ON credit_grants(payment_intent_id);
CREATE INDEX IF NOT EXISTS credit_grants_ch ON credit_grants(charge_id);
CREATE INDEX IF NOT EXISTS credit_grants_cs ON credit_grants(checkout_session_id);
CREATE INDEX IF NOT EXISTS credit_grants_event ON credit_grants(stripe_event_id);

CREATE TABLE IF NOT EXISTS stripe_disputes (
  dispute_id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  status TEXT NOT NULL,
  charge_id TEXT,
  payment_intent_id TEXT,
  stripe_event_id TEXT,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch())
);
