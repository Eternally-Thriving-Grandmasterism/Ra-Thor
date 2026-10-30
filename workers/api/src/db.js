// D1-backed balances, ledger and holds. Every balance mutation is either a single
// conditional SQL statement or a D1 batch (batches run as one transaction).

// PRICING-4 launch offer: NO free signup credits. New accounts start at 0; the only way in is
// the paid $1 trial (STRIPE_PRICE_TRIAL, 1,000 credits, once per account) or a paid pack/plan.
export const SIGNUP_GRANT_CREDITS = 0;

// Create the account (0 credits) if it does not exist.
export async function ensureAccount(db, id) {
  return db.prepare('INSERT OR IGNORE INTO accounts (id, balance) VALUES (?1, ?2)').bind(id, SIGNUP_GRANT_CREDITS).run();
}

export async function available(db, id) {
  const r = await db.prepare(
    "SELECT a.balance - COALESCE((SELECT SUM(amount) FROM holds h WHERE h.account_id = a.id AND h.status = 'open'), 0) AS avail FROM accounts a WHERE a.id = ?1",
  ).bind(id).first();
  return r ? r.avail : 0;
}

// Atomically place a hold only if the account is not frozen and available
// credits cover it. Returns true if held.
export async function placeHold(db, holdId, id, amount) {
  const r = await db.prepare(
    `INSERT INTO holds (id, account_id, amount, status)
     SELECT ?1, ?2, ?3, 'open' WHERE (
       SELECT a.balance - COALESCE((SELECT SUM(amount) FROM holds h WHERE h.account_id = a.id AND h.status = 'open'), 0)
       FROM accounts a WHERE a.id = ?2 AND COALESCE(a.frozen, 0) = 0 AND COALESCE(a.settle_locked, 0) = 0) >= ?3`,
  ).bind(holdId, id, amount).run();
  return (r.meta?.changes ?? 0) === 1;
}

// Settle a hold at the actual charge and close it, in one transaction.
// HOLD-HEADROOM-1: if actual > hold, the overage is debited from the balance that is not
// reserved by OTHER open holds (so their later settles cannot fail). Whatever cannot be
// covered is recorded in settle_shortfalls (migration 0008) plus a 0-delta ledger row, the
// account gets settle_locked = 1 (FREEZE-PRECEDENCE-1: its own flag, never freeze_reason, so it
// cannot hide or overwrite a dispute freeze; placeHold refuses frozen or settle_locked accounts),
// and a settle_over_hold alert is logged. Returns the credits actually debited.
export async function settleHold(db, holdId, actual, meta = {}) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const h = await db.prepare("SELECT account_id, amount FROM holds WHERE id = ?1 AND status = 'open'").bind(holdId).first();
    if (!h) return null;
    if (actual <= h.amount) {
      const charge = actual;
      await db.batch([
        db.prepare("UPDATE holds SET status = 'settled' WHERE id = ?1 AND status = 'open'").bind(holdId),
        db.prepare('UPDATE accounts SET balance = balance - ?2 WHERE id = ?1 AND changes() = 1').bind(h.account_id, charge),
        db.prepare("INSERT INTO ledger (account_id, delta, reason, hold_id) SELECT ?1, ?2, 'chat', ?3 WHERE changes() = 1").bind(h.account_id, -charge, holdId),
      ]);
      return charge;
    }
    const a = await db.prepare(
      `SELECT a.balance AS balance, COALESCE((SELECT SUM(amount) FROM holds o WHERE o.account_id = a.id AND o.status = 'open' AND o.id <> ?2), 0) AS other
       FROM accounts a WHERE a.id = ?1`,
    ).bind(h.account_id, holdId).first();
    const free = Math.max(0, (a?.balance ?? 0) - (a?.other ?? 0) - h.amount);
    const over = actual - h.amount;
    const extra = Math.min(over, free);
    const charge = h.amount + extra;
    const shortfall = actual - charge;
    const stmts = [
      // CAS on the balance we read (and the hold still open), so a concurrent change makes
      // this whole batch a no-op and we retry instead of overdrawing.
      db.prepare("UPDATE accounts SET balance = balance - ?2 WHERE id = ?1 AND balance = ?3 AND EXISTS (SELECT 1 FROM holds WHERE id = ?4 AND status = 'open')").bind(h.account_id, charge, a.balance, holdId),
      db.prepare("UPDATE holds SET status = 'settled' WHERE id = ?1 AND status = 'open' AND changes() = 1").bind(holdId),
      db.prepare("INSERT INTO ledger (account_id, delta, reason, hold_id) SELECT ?1, ?2, 'chat', ?3 WHERE changes() = 1").bind(h.account_id, -charge, holdId),
    ];
    if (shortfall > 0) {
      stmts.push(
        db.prepare("INSERT INTO settle_shortfalls (account_id, hold_id, actual_credits, charged_credits, shortfall_credits, model) SELECT ?1, ?2, ?3, ?4, ?5, ?6 WHERE changes() = 1")
          .bind(h.account_id, holdId, actual, charge, shortfall, meta.model ?? null),
        db.prepare("INSERT INTO ledger (account_id, delta, reason, hold_id) SELECT ?1, 0, 'settle_shortfall', ?2 WHERE changes() = 1").bind(h.account_id, holdId),
        db.prepare('UPDATE accounts SET settle_locked = 1 WHERE id = ?1 AND changes() = 1').bind(h.account_id),
      );
    }
    await db.batch(stmts);
    const led = await db.prepare("SELECT COUNT(*) AS n FROM ledger WHERE hold_id = ?1 AND reason = 'chat'").bind(holdId).first();
    if (led?.n > 0) {
      // eslint-disable-next-line no-console
      console.error(JSON.stringify({ msg: 'settle_over_hold', alert: shortfall > 0, account_id: h.account_id, hold_id: holdId, model: meta.model ?? null, held: h.amount, actual, charged: charge, shortfall, frozen: shortfall > 0 }));
      return charge;
    }
  }
  throw new Error('settle_over_hold_cas_failed');
}

// API-ABUSE-HARDEN-1: called once xAI usage has come back. Retries settleHold with backoff; if
// it still fails, charges exactly `actual` (never more). If that write fails too, records the whole
// amount as a shortfall and sets settle_locked. The hold is never released silently.
export async function settleWithFallback(db, holdId, actual, meta = {}, backoff = [25, 100, 400]) {
  for (let i = 0; ; i += 1) {
    try { return await settleHold(db, holdId, actual, meta); } catch (e) {
      if (i >= backoff.length) break;
      await new Promise((r) => setTimeout(r, backoff[i]));
    }
  }
  const h = await db.prepare("SELECT account_id FROM holds WHERE id = ?1 AND status = 'open'").bind(holdId).first();
  if (!h) return null;
  const log = (msg, charged) => console.error(JSON.stringify({ msg, alert: true, account_id: h.account_id, hold_id: holdId, model: meta.model ?? null, actual, charged }));
  try {
    await db.batch([
      db.prepare("UPDATE holds SET status = 'settled' WHERE id = ?1 AND status = 'open'").bind(holdId),
      db.prepare('UPDATE accounts SET balance = balance - ?2 WHERE id = ?1 AND changes() = 1').bind(h.account_id, actual),
      db.prepare("INSERT INTO ledger (account_id, delta, reason, hold_id) SELECT ?1, ?2, 'chat', ?3 WHERE changes() = 1").bind(h.account_id, -actual, holdId),
    ]);
    log('settle_fallback_exact_charge', actual);
    return actual;
  } catch { /* fall through to shortfall */ }
  try {
    await db.batch([
    db.prepare("UPDATE holds SET status = 'settled' WHERE id = ?1 AND status = 'open'").bind(holdId),
    db.prepare('INSERT INTO settle_shortfalls (account_id, hold_id, actual_credits, charged_credits, shortfall_credits, model) SELECT ?1, ?2, ?3, 0, ?3, ?4 WHERE changes() = 1 AND ?3 > 0')
      .bind(h.account_id, holdId, actual, meta.model ?? null),
    db.prepare("INSERT INTO ledger (account_id, delta, reason, hold_id) SELECT ?1, 0, 'settle_shortfall', ?2 WHERE changes() = 1").bind(h.account_id, holdId),
    db.prepare('UPDATE accounts SET settle_locked = 1 WHERE id = ?1').bind(h.account_id),
    ]);
  } catch (e) {
    // D1 is down: the hold stays OPEN (reserved, not released) and this alert needs a human.
    log('settle_unrecorded_hold_left_open', 0);
    throw e;
  }
  log('settle_fallback_shortfall', 0);
  return 0;
}

export async function releaseHold(db, holdId) {
  await db.prepare("UPDATE holds SET status = 'released' WHERE id = ?1 AND status = 'open'").bind(holdId).run();
}

export async function accountForKeyHash(db, keyHash) {
  return db.prepare('SELECT key_hash, account_id FROM api_keys WHERE key_hash = ?1 AND revoked = 0').bind(keyHash).first();
}

// Record the Stripe event and apply its grant in one transaction. The UNIQUE constraint on
// stripe_events.event_id makes a replay fail the whole batch => returns { duplicate: true }.
export async function applyStripeEvent(db, event, stmts) {
  try {
    await db.batch([db.prepare('INSERT INTO stripe_events (event_id, type) VALUES (?1, ?2)').bind(event.id, event.type), ...stmts]);
    return { duplicate: false };
  } catch (e) {
    const m = String(e?.message || e);
    // stripe_events is the first statement, so a replay always reports duplicate, even for a trial.
    if (/UNIQUE/i.test(m) && /trial_redemptions/i.test(m)) return { duplicate: false, trialUsed: true };
    if (/UNIQUE/i.test(m)) return { duplicate: true };
    throw e;
  }
}

export const grantStmts = (db, id, credits, reason, eventId, extra = {}) => [
  db.prepare('INSERT OR IGNORE INTO accounts (id, balance) VALUES (?1, 0)').bind(id),
  db.prepare('UPDATE accounts SET balance = balance + ?2, stripe_customer_id = COALESCE(?3, stripe_customer_id) WHERE id = ?1').bind(id, credits, extra.customer ?? null),
  db.prepare('INSERT INTO ledger (account_id, delta, reason, stripe_event_id) VALUES (?1, ?2, ?3, ?4)').bind(id, credits, reason, eventId),
  db.prepare(
    `INSERT INTO credit_grants (account_id, stripe_event_id, payment_intent_id, charge_id, checkout_session_id, granted_credits, amount_paid)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)`,
  ).bind(id, eventId, extra.paymentIntent ?? null, extra.charge ?? null, extra.checkoutSession ?? null, credits, extra.amountPaid ?? 0),
];

// Stripe expandable ids arrive as a string or {id}.
export function stripeId(v) {
  if (typeof v === 'string' && v.length > 0) return v;
  if (v && typeof v === 'object' && typeof v.id === 'string' && v.id.length > 0) return v.id;
  return null;
}

// Mechanical claw target (REFUND-1). ceil(granted × refunded / paid).
// Full refund (refunded >= paid) or a missing/zero paid amount → the whole grant.
export function clawTarget(granted, amountRefunded, amountPaid) {
  if (!Number.isInteger(granted) || granted <= 0) return 0;
  if (!Number.isInteger(amountRefunded) || amountRefunded <= 0) return 0;
  if (!Number.isInteger(amountPaid) || amountPaid <= 0) return granted;
  if (amountRefunded >= amountPaid) return granted;
  return Math.ceil((granted * amountRefunded) / amountPaid);
}

export const DISPUTE_UNFREEZE_STATUSES = Object.freeze(['won', 'warning_closed']);
// Closed as lost: claw like a full refund and stay frozen until a human clears accounts.frozen.
export const DISPUTE_LOST_STATUSES = Object.freeze(['lost', 'charge_refunded']);
export const DISPUTE_TERMINAL_STATUSES = Object.freeze([...DISPUTE_UNFREEZE_STATUSES, ...DISPUTE_LOST_STATUSES]);
export const isDisputeUnfreezeStatus = (s) => DISPUTE_UNFREEZE_STATUSES.includes(String(s || ''));
export const isDisputeLostStatus = (s) => DISPUTE_LOST_STATUSES.includes(String(s || ''));
export const isDisputeTerminalStatus = (s) => DISPUTE_TERMINAL_STATUSES.includes(String(s || ''));

const OPEN_HOLDS = `COALESCE((SELECT SUM(amount) FROM holds h WHERE h.account_id = a.id AND h.status = 'open'), 0)`;

export async function findGrantById(db, id) {
  if (id == null) return null;
  return db.prepare('SELECT * FROM credit_grants WHERE id = ?1').bind(id).first();
}

export async function findGrant(db, ids = {}) {
  const ch = stripeId(ids.charge);
  const pi = stripeId(ids.paymentIntent);
  const cs = stripeId(ids.checkoutSession);
  if (ch) {
    const r = await db.prepare('SELECT * FROM credit_grants WHERE charge_id = ?1 ORDER BY id DESC LIMIT 1').bind(ch).first();
    if (r) return r;
  }
  if (pi) {
    const r = await db.prepare('SELECT * FROM credit_grants WHERE payment_intent_id = ?1 ORDER BY id DESC LIMIT 1').bind(pi).first();
    if (r) return r;
  }
  if (cs) {
    const r = await db.prepare('SELECT * FROM credit_grants WHERE checkout_session_id = ?1 ORDER BY id DESC LIMIT 1').bind(cs).first();
    if (r) return r;
  }
  return null;
}

export async function findAccountByCustomer(db, customer) {
  const id = stripeId(customer);
  if (!id) return null;
  const r = await db.prepare('SELECT id FROM accounts WHERE stripe_customer_id = ?1').bind(id).first();
  return r?.id ?? null;
}

export async function isFrozen(db, id) {
  const r = await db.prepare('SELECT frozen, settle_locked FROM accounts WHERE id = ?1').bind(id).first();
  return !!(r && (r.frozen || r.settle_locked));
}

export async function accountBalance(db, id) {
  const r = await db.prepare('SELECT balance, frozen, freeze_reason, settle_locked FROM accounts WHERE id = ?1').bind(id).first();
  return r || { balance: 0, frozen: 0, freeze_reason: null, settle_locked: 0 };
}

// Keep a shortfall freeze; do not overwrite it with a later dispute reason.
// FREEZE-PRECEDENCE-1: settle_over_hold is no longer a freeze_reason (see settle_locked), so a
// later dispute always lands in this slot. A legacy 'settle_over_hold' value is overwritten.
export const freezeStmt = (db, id, reason) =>
  db.prepare(
    `UPDATE accounts SET frozen = 1, freeze_reason = CASE
       WHEN freeze_reason IN ('refund_claw_shortfall') THEN freeze_reason
       ELSE ?2
     END WHERE id = ?1`,
  ).bind(id, reason);

// Unfreeze a dispute freeze only when no stripe_disputes row for this account
// is still open (status not in won / warning_closed). lost / charge_refunded
// count as still blocking. Call after upserting the closing row.
// FREEZE-PRECEDENCE-1: unlock a settle_over_hold lock. Touches ONLY settle_locked and the
// account's unresolved settle_shortfalls; frozen / freeze_reason (dispute, dispute_lost,
// refund_claw_shortfall) are never changed, so an active dispute freeze stays in place.
// Refuses while an UNAPPLIED early dispute (applied_account_id NULL) matches one of this
// account's credit_grants by payment_intent_id or charge_id and is not won/warning_closed:
// that dispute is not yet visible as a freeze on the account.
export async function pendingEarlyDisputes(db, id) {
  const r = await db.prepare(
    `SELECT e.dispute_id, e.status FROM early_disputes e
     WHERE e.applied_account_id IS NULL AND e.status NOT IN ('won', 'warning_closed')
       AND EXISTS (SELECT 1 FROM credit_grants g WHERE g.account_id = ?1
         AND ((e.payment_intent_id IS NOT NULL AND g.payment_intent_id = e.payment_intent_id)
           OR (e.charge_id IS NOT NULL AND g.charge_id = e.charge_id)))`,
  ).bind(id).all();
  return r.results || [];
}
export async function unlockSettleOverHold(db, id) {
  const pending = await pendingEarlyDisputes(db, id);
  if (pending.length) return { unlocked: false, reason: 'unapplied_early_dispute', disputes: pending.map((p) => p.dispute_id) };
  await db.batch([
    db.prepare('UPDATE settle_shortfalls SET resolved = 1 WHERE account_id = ?1 AND resolved = 0').bind(id),
    db.prepare('UPDATE accounts SET settle_locked = 0 WHERE id = ?1').bind(id),
  ]);
  const a = await accountBalance(db, id);
  return { unlocked: true, still_frozen: !!a.frozen, freeze_reason: a.freeze_reason ?? null };
}

export const unfreezeIfNoOpenDisputesStmt = (db, id) =>
  db.prepare(
    `UPDATE accounts SET frozen = 0, freeze_reason = NULL
     WHERE id = ?1 AND freeze_reason IN ('dispute')
       AND NOT EXISTS (
         SELECT 1 FROM stripe_disputes d
         WHERE d.account_id = ?1 AND d.status NOT IN ('won', 'warning_closed')
       )`,
  ).bind(id);

// Compare-and-set claw. Grant row first (WHERE clawed_credits = already, and
// available covers actual). Balance, then ledger, chain on changes()=1 so
// neither ledger nor a failed chain can stick without the balance change.
export function clawCasStmts(db, { grantId, accountId, already, actual, reason, eventId }) {
  const stmts = [
    db.prepare(
      `UPDATE credit_grants SET clawed_credits = clawed_credits + ?2
       WHERE id = ?1 AND clawed_credits = ?3
         AND (?2 = 0 OR (
           SELECT a.balance - ${OPEN_HOLDS}
           FROM accounts a WHERE a.id = credit_grants.account_id
         ) >= ?2)`,
    ).bind(grantId, actual, already),
  ];
  if (actual > 0) {
    stmts.push(db.prepare(
      `UPDATE accounts SET balance = balance - ?2
       WHERE id = ?1 AND changes() = 1
         AND (balance - COALESCE((SELECT SUM(amount) FROM holds h WHERE h.account_id = accounts.id AND h.status = 'open'), 0)) >= ?2`,
    ).bind(accountId, actual));
    stmts.push(db.prepare(
      `INSERT INTO ledger (account_id, delta, reason, stripe_event_id)
       SELECT ?1, ?2, ?3, ?4 WHERE changes() = 1`,
    ).bind(accountId, -actual, reason, eventId));
  }
  return stmts;
}

export async function applyClawCas(db, { grantId, accountId, already, actual, reason, eventId }) {
  const out = await db.batch(clawCasStmts(db, { grantId, accountId, already, actual, reason, eventId }));
  if ((out[0]?.meta?.changes ?? 0) !== 1) return { conflict: true };
  if (actual > 0) {
    if ((out[1]?.meta?.changes ?? 0) !== 1 || (out[2]?.meta?.changes ?? 0) !== 1) return { conflict: true };
  }
  return { ok: true };
}

export const unmatchedStmt = (db, event, reason, d = {}) => {
  const it = d.item || {};
  const amount = Number.isInteger(d.amount) ? d.amount : null;
  return db.prepare(
    `INSERT OR IGNORE INTO unmatched_payments (stripe_event_id, event_type, reason, account_id, price_id, amount, currency, mode, metadata_credits, detail)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)`,
  ).bind(
    event.id,
    event.type,
    reason,
    d.user ?? null,
    it.priceId ?? null,
    amount,
    (it.priceCurrency ?? it.currency ?? d.currency) ?? null,
    d.mode ?? null,
    d.metadata?.credits != null ? String(d.metadata.credits) : null,
    d.detail != null ? String(d.detail) : null,
  );
};

export const disputeUpsertStmt = (db, disputeId, accountId, status, chargeId, paymentIntentId, eventId) =>
  db.prepare(
    `INSERT INTO stripe_disputes (dispute_id, account_id, status, charge_id, payment_intent_id, stripe_event_id)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6)
     ON CONFLICT(dispute_id) DO UPDATE SET
       status = CASE
         WHEN stripe_disputes.status IN ('won', 'warning_closed', 'lost', 'charge_refunded') THEN stripe_disputes.status
         ELSE excluded.status
       END,
       stripe_event_id = excluded.stripe_event_id,
       updated_at = unixepoch()`,
  ).bind(disputeId, accountId, status, chargeId, paymentIntentId, eventId);

// SPROOF-FIX-DISPUTE-ORDER: store a dispute that has no grant yet. Idempotent on
// dispute_id; terminal statuses are sticky. Keyed by payment_intent and charge.
export const earlyDisputeUpsertStmt = (db, d) =>
  db.prepare(
    `INSERT INTO early_disputes (dispute_id, status, payment_intent_id, charge_id, amount, currency, stripe_event_id)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
     ON CONFLICT(dispute_id) DO UPDATE SET
       status = CASE
         WHEN early_disputes.status IN ('won', 'warning_closed', 'lost', 'charge_refunded') THEN early_disputes.status
         ELSE excluded.status
       END,
       payment_intent_id = COALESCE(early_disputes.payment_intent_id, excluded.payment_intent_id),
       charge_id = COALESCE(early_disputes.charge_id, excluded.charge_id),
       amount = COALESCE(excluded.amount, early_disputes.amount),
       currency = COALESCE(excluded.currency, early_disputes.currency),
       stripe_event_id = excluded.stripe_event_id,
       updated_at = unixepoch()`,
  ).bind(d.disputeId, d.status, d.paymentIntent ?? null, d.charge ?? null, Number.isInteger(d.amount) ? d.amount : null, d.currency ?? null, d.eventId);

// Unapplied early disputes matching this grant's PI or charge (NULLs never match).
const EARLY_MATCH = `e.applied_event_id IS NULL AND ((?2 IS NOT NULL AND e.payment_intent_id = ?2) OR (?3 IS NOT NULL AND e.charge_id = ?3))`;
const LOST_SQL = `('lost', 'charge_refunded')`;
const CLOSED_SQL = `('won', 'warning_closed', 'lost', 'charge_refunded')`;

// SPROOF-FIX-DISPUTE-ORDER: statements appended to the grant batch, right after the
// credit_grants INSERT (same transaction). ?1 account, ?2 payment_intent, ?3 charge,
// ?4 grant event id. Open early dispute → freeze 'dispute' (like dispute.created).
// Lost → claw the whole grant capped at available, freeze 'dispute_lost' (or keep
// 'refund_claw_shortfall' + unmatched row on shortfall), like the dispute.closed lost
// path. Won / warning_closed → no freeze. Every matched dispute is copied into
// stripe_disputes (so a later dispute.closed won can unfreeze) and marked applied.
export const earlyDisputeGrantStmts = (db, accountId, paymentIntent, charge, eventId) => {
  const args = [accountId, paymentIntent ?? null, charge ?? null, eventId];
  const lost = `EXISTS (SELECT 1 FROM early_disputes e WHERE ${EARLY_MATCH} AND e.status IN ${LOST_SQL})`;
  const open = `EXISTS (SELECT 1 FROM early_disputes e WHERE ${EARLY_MATCH} AND e.status NOT IN ${CLOSED_SQL})`;
  const g = `(SELECT * FROM credit_grants WHERE stripe_event_id = ?4 AND account_id = ?1 ORDER BY id DESC LIMIT 1)`;
  return [
    // Lost claw on the grant row: min(granted, available) where available excludes open holds.
    db.prepare(
      `UPDATE credit_grants SET clawed_credits = MIN(granted_credits, MAX(0, (
         SELECT a.balance - ${OPEN_HOLDS} FROM accounts a WHERE a.id = ?1)))
       WHERE id = (SELECT id FROM ${g}) AND ${lost}`,
    ).bind(...args),
    db.prepare(
      `UPDATE accounts SET balance = balance - (SELECT clawed_credits FROM ${g})
       WHERE id = ?1 AND ${lost}`,
    ).bind(...args),
    db.prepare(
      `INSERT INTO ledger (account_id, delta, reason, stripe_event_id)
       SELECT ?1, -clawed_credits, 'dispute_lost_claw', ?4 FROM ${g}
       WHERE clawed_credits > 0 AND ${lost}`,
    ).bind(...args),
    db.prepare(
      `INSERT OR IGNORE INTO unmatched_payments (stripe_event_id, event_type, reason, account_id, amount, currency, mode, detail)
       SELECT ?4, 'early_dispute_at_grant', 'refund_claw_shortfall', ?1, gr.amount_paid, NULL, 'charge', gr.granted_credits - gr.clawed_credits
       FROM ${g} gr WHERE gr.clawed_credits < gr.granted_credits AND ${lost}`,
    ).bind(...args),
    db.prepare(
      `UPDATE accounts SET frozen = 1, freeze_reason = CASE
         WHEN freeze_reason IN ('refund_claw_shortfall') THEN freeze_reason
         WHEN (SELECT clawed_credits < granted_credits FROM ${g}) THEN 'refund_claw_shortfall'
         ELSE 'dispute_lost'
       END WHERE id = ?1 AND ${lost}`,
    ).bind(...args),
    db.prepare(
      `UPDATE accounts SET frozen = 1, freeze_reason = CASE
         WHEN freeze_reason IN ('refund_claw_shortfall', 'dispute_lost') THEN freeze_reason
         ELSE 'dispute'
       END WHERE id = ?1 AND ${open}`,
    ).bind(...args),
    db.prepare(
      `INSERT OR IGNORE INTO stripe_disputes (dispute_id, account_id, status, charge_id, payment_intent_id, stripe_event_id)
       SELECT e.dispute_id, ?1, e.status, e.charge_id, e.payment_intent_id, e.stripe_event_id
       FROM early_disputes e WHERE ${EARLY_MATCH} AND ?4 IS NOT NULL`,
    ).bind(...args),
    db.prepare(
      `UPDATE early_disputes SET applied_event_id = ?4, applied_account_id = ?1, updated_at = unixepoch()
       WHERE dispute_id IN (SELECT e.dispute_id FROM early_disputes e WHERE ${EARLY_MATCH})`,
    ).bind(...args),
  ];
};

// PRICING-3: claims the one paid trial for this account in the same batch as the grant.
export const trialClaimStmt = (db, id, eventId) =>
  db.prepare('INSERT INTO trial_redemptions (account_id, stripe_event_id) VALUES (?1, ?2)').bind(id, eventId);

export async function accountExists(db, id) {
  return !!(await db.prepare('SELECT 1 AS ok FROM accounts WHERE id = ?1').bind(id).first());
}

export async function trialAlreadyUsed(db, id) {
  return !!(await db.prepare('SELECT 1 AS ok FROM trial_redemptions WHERE account_id = ?1').bind(id).first());
}

// D1/SQLite reports a missing table as "no such table: <name>".
export const isSchemaMissing = (e) => /no such table/i.test(String(e?.message || e));
