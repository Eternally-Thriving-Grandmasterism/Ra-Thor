-- FREEZE-PRECEDENCE-1: additive. settle_over_hold moves out of the single freeze_reason slot
-- into its own flag, so it can never hide or overwrite a dispute / dispute_lost freeze.
-- Do not edit 0001-0008.
ALTER TABLE accounts ADD COLUMN settle_locked INTEGER NOT NULL DEFAULT 0;

-- Backfill 1: every account frozen for settle_over_hold, or with an unresolved shortfall,
-- gets the new lock (locking more is safe; nothing is unlocked by this).
UPDATE accounts SET settle_locked = 1
 WHERE freeze_reason = 'settle_over_hold'
    OR id IN (SELECT account_id FROM settle_shortfalls WHERE resolved = 0);

-- Backfill 2: restore any freeze that settle_over_hold hid. Order: refund_claw_shortfall
-- (unmatched_payments row with that reason for the account), then dispute_lost, then dispute
-- (from stripe_disputes), else clear. settle_locked (set above) keeps the account locked.
UPDATE accounts SET
  freeze_reason = CASE
    WHEN EXISTS (SELECT 1 FROM unmatched_payments u WHERE u.account_id = accounts.id AND u.reason = 'refund_claw_shortfall') THEN 'refund_claw_shortfall'
    WHEN EXISTS (SELECT 1 FROM stripe_disputes d WHERE d.account_id = accounts.id AND d.status IN ('lost', 'charge_refunded')) THEN 'dispute_lost'
    WHEN EXISTS (SELECT 1 FROM stripe_disputes d WHERE d.account_id = accounts.id AND d.status NOT IN ('won', 'warning_closed')) THEN 'dispute'
    ELSE NULL END,
  frozen = CASE
    WHEN EXISTS (SELECT 1 FROM unmatched_payments u WHERE u.account_id = accounts.id AND u.reason = 'refund_claw_shortfall') THEN 1
    WHEN EXISTS (SELECT 1 FROM stripe_disputes d WHERE d.account_id = accounts.id AND d.status NOT IN ('won', 'warning_closed')) THEN 1
    ELSE 0 END
 WHERE freeze_reason = 'settle_over_hold';
