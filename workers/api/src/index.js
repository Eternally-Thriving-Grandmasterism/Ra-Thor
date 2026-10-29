// Ra-Thor API Worker for api.rathor.ai — SANDBOX DRAFT, not deployed.
// Seller of hosted credits: Autonomicity Games Inc.
// All secrets come from Worker secrets; none live in this repo.
// Worker secrets (wrangler secret put): XAI_API_KEY, STRIPE_WEBHOOK_SECRET, STRIPE_SECRET_KEY
// (key prefix must match LIVE_MODE; used only to GET checkout line items), RESEND_API_KEY.
import { creditsForSettle, holdEstimate, ratesVerified, PRICING } from './credits.js';
import { resolveLineItem, checkoutLineItem, invoiceLineItem, resolveCheckoutPrice } from './prices.js';
import { verifyStripeSignature } from './stripe.js';
import { authenticateBearer } from './auth.js';
import { ALLOWED_MODELS, DEFAULT_MODEL, MAX_TOKENS_CAP, DEFAULT_MAX_TOKENS, CHAT_MESSAGES_MAX_BYTES, CSRF_HEADER, isLiveMode, stripeKeyMatchesLiveMode, stripeRetryExhaustSeconds } from './config.js';
import { mercyChecks, rebuildMessages, mercyLevelGate, recordBlock, RULES_VERSION } from './mercyChecks.js';
import { ensureAccount, placeHold, settleHold, releaseHold, available, applyStripeEvent, grantStmts, trialClaimStmt, accountExists, isSchemaMissing, trialAlreadyUsed, stripeId, clawTarget, findGrant, findGrantById, findAccountByCustomer, isFrozen, accountBalance, freezeStmt, unfreezeIfNoOpenDisputesStmt, unmatchedStmt, disputeUpsertStmt, earlyDisputeUpsertStmt, earlyDisputeGrantStmts, applyClawCas, isDisputeUnfreezeStatus, isDisputeTerminalStatus } from './db.js';
import { requestMagicLink, verifyMagicLink, verifyConfirmPage, logout, issueKey, revokeKeys, requireSessionCsrf, authenticateSession, csrfGuard, parseCookie } from './session.js';

const XAI_URL = 'https://api.x.ai/v1/chat/completions';
const STRIPE_API = 'https://api.stripe.com/v1';
const RESEND_URL = 'https://api.resend.com/emails';
export const RECEIPT_FROM = 'Ra-Thor <info@rathor.ai>';
const enc = new TextEncoder();

function cors(env, req) {
  const allowed = env.ALLOWED_ORIGIN || 'https://rathor.ai';
  const h = {
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': `Content-Type, Authorization, ${CSRF_HEADER}`,
    Vary: 'Origin',
  };
  // Credentials only from https://rathor.ai (or ALLOWED_ORIGIN). Never '*'.
  if (req.headers.get('Origin') === allowed) {
    h['Access-Control-Allow-Origin'] = allowed;
    h['Access-Control-Allow-Credentials'] = 'true';
  }
  return h;
}
const json = (obj, status, headers = {}) => new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json', ...headers } });

// Validates and normalises the client body. Returns { error } or { body }.
export const UPSTREAM_BODY_FIELDS = Object.freeze(['model', 'messages', 'max_tokens', 'stream', 'temperature']);
export function validateChat(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return { error: 'bad_body' };
  if (!Array.isArray(body.messages)) return { error: 'messages_required' };
  const model = body.model ?? DEFAULT_MODEL;
  if (!ALLOWED_MODELS.includes(model)) return { error: 'model_not_allowed', allowed: ALLOWED_MODELS };
  let mt = body.max_tokens ?? DEFAULT_MAX_TOKENS;
  if (!Number.isInteger(mt) || mt < 1) return { error: 'max_tokens_out_of_range', max: MAX_TOKENS_CAP };
  if (mt > MAX_TOKENS_CAP) mt = MAX_TOKENS_CAP;
  // Forward only known-safe fields (no client-chosen n, tools, stream, etc.).
  // MERCY-LAUNCH-1: messages are rebuilt from an allowlist; anything else is 400 unsupported_content.
  const rb = rebuildMessages(body.messages);
  if (rb.error) return { error: rb.error };
  // MERCY-SAFETY-SCOPE-2: the upstream body is built only from these fields. Any other client
  // field (for example a safety, moderation or filter setting) is dropped, and the Worker adds
  // no field of its own that could loosen or turn off the provider's default safety.
  const out = { model, messages: rb.messages, max_tokens: mt, stream: false };
  if (typeof body.temperature === 'number' && Number.isFinite(body.temperature)) out.temperature = body.temperature;
  return { body: out };
}

async function chat(req, env, h) {
  const allowed = env.ALLOWED_ORIGIN || 'https://rathor.ai';
  const byKey = await authenticateBearer(req, env);
  let user;
  if (byKey) {
    // Bearer: unchanged. Origin check only; no CSRF header.
    if (req.headers.get('Origin') !== allowed) return json({ error: 'origin_not_allowed' }, 403, h);
    user = byKey;
  } else if (parseCookie(req)) {
    const csrf = csrfGuard(req, env);
    if (csrf) return json({ error: csrf.error }, csrf.status, h);
    user = await authenticateSession(req, env);
    if (!user) return json({ error: 'unauthorized' }, 401, h);
  } else {
    if (req.headers.get('Origin') !== allowed) return json({ error: 'origin_not_allowed' }, 403, h);
    return json({ error: 'unauthorized' }, 401, h);
  }
  // Chat does not call Stripe. A missing or mismatched STRIPE_SECRET_KEY must not 500 here.
  // TODO(Sherif): re-check xAI rates on docs.x.ai, then flip RATES_VERIFIED; billing is refused until RATES_VERIFIED is "true".
  if (!ratesVerified(env)) return json({ error: 'billing_disabled_rates_unverified', pricing_status: PRICING.status }, 503, h);
  // MERCY-LAUNCH-1: production refuses chat unless MERCY_CHECKS_LEVEL is exactly 'launch'. Before the hold; never charged.
  const lv = mercyLevelGate(env);
  if (lv) return json(lv, 503, h);
  const cl = Number(req.headers.get('Content-Length'));
  if (Number.isFinite(cl) && cl > CHAT_MESSAGES_MAX_BYTES + 8192) return json({ error: 'payload_too_large', max_bytes: CHAT_MESSAGES_MAX_BYTES }, 413, h);
  let raw;
  try { raw = await req.json(); } catch { return json({ error: 'bad_json' }, 400, h); }
  const v = validateChat(raw);
  if (v.error) return json(v, 400, h);
  if (enc.encode(JSON.stringify(v.body.messages)).length > CHAT_MESSAGES_MAX_BYTES) {
    return json({ error: 'payload_too_large', max_bytes: CHAT_MESSAGES_MAX_BYTES }, 413, h);
  }
  const mc = mercyChecks.checkRequest(v.body);
  if (!mc.allowed) {
    // No ensureAccount: account_id is null and nothing is charged.
    await recordBlock(env.DB, { stage: 'request', ruleId: mc.reason, model: v.body.model, accountId: null, chargedCredits: 0 });
    return json({ error: 'mercy_checks_denied', stage: 'request', reason: mc.reason, rules_version: RULES_VERSION }, 400, h);
  }

  await ensureAccount(env.DB, user);
  if (await isFrozen(env.DB, user)) return json({ error: 'account_frozen' }, 403, h);
  // HOLD-HEADROOM-1: prompt bytes + per-request/per-message overhead + max_tokens + reasoning headroom.
  const hold = holdEstimate(v.body.model, enc.encode(JSON.stringify(v.body.messages)).length, v.body.messages.length, v.body.max_tokens);
  const holdId = crypto.randomUUID();
  if (!(await placeHold(env.DB, holdId, user, hold))) return json({ error: 'insufficient_credits', required_hold: hold }, 402, h);

  // From here on the hold is always closed: settled on success, released on any failure
  // (fetch, non-2xx, r.json(), or settleHold throwing). releaseHold is a no-op once settled.
  try {
    let r;
    try {
      r = await fetch(XAI_URL, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.XAI_API_KEY}` }, body: JSON.stringify(v.body) });
    } catch { return json({ error: 'upstream_unreachable' }, 502, h); }
    if (!r.ok) return json({ error: 'upstream_error', status: r.status }, 502, h);
    let data;
    try { data = await r.json(); } catch { return json({ error: 'upstream_bad_json' }, 502, h); }
    const charged = await settleHold(env.DB, holdId, creditsForSettle(v.body.model, data.usage, hold), { model: v.body.model });
    const remaining = await available(env.DB, user);
    const meta = { credits_held: hold, credits_charged: charged, credits_remaining: remaining, pricing_status: PRICING.status };
    // Tokens were used, so the charge stands even if Mercy-Checks withholds the reply.
    const out = mercyChecks.checkResponse(data);
    // Every text field is withheld on a block; an unknown reply shape fails closed here too.
    if (!out.allowed) {
      await recordBlock(env.DB, { stage: 'response', ruleId: out.reason, model: v.body.model, accountId: user, chargedCredits: charged });
      return json({ error: 'mercy_checks_denied', stage: 'response', reason: out.reason, rules_version: RULES_VERSION, rathor: meta }, 422, h);
    }
    return json({ ...out.shaped, rathor: meta }, 200, h);
  } finally {
    try { await releaseHold(env.DB, holdId); } catch { /* best effort; hold stays open only if D1 itself is down */ }
  }
}

async function sendReceipt(env, to, credits) {
  if (!env.RESEND_API_KEY || !to) return false;
  const r = await fetch(RESEND_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.RESEND_API_KEY}` },
    body: JSON.stringify({ from: RECEIPT_FROM, to: [to], subject: 'Ra-Thor credits receipt (test mode)', text: `Test-mode receipt: ${credits} credits were added to your Ra-Thor account.` }),
  });
  return r.ok;
}

const ACTIVE = new Set(['active', 'trialing']);

// Metadata is client-influenced; it is never used to grant. A disagreement is only logged.
function logMetadataMismatch(event, metaCredits, granted) {
  if (metaCredits === undefined || metaCredits === null || metaCredits === '') return;
  if (String(metaCredits) !== String(granted)) console.warn(JSON.stringify({ msg: 'stripe_metadata_credits_mismatch', event_id: event.id, metadata_credits: String(metaCredits), granted }));
}

const ALERT_UNMATCHED = new Set([
  'livemode_mismatch',
  'refund_no_matching_grant',
  'refund_claw_shortfall',
  'dispute_no_matching_grant',
  'renewal_payment_ids_missing',
]);

// Unknown / non-standard payment (incl. unknown or missing user, second trial): record it
// (idempotently, with the event id) and answer 200 so Stripe stops retrying. No credits are granted.
async function refuse(db, event, reason, d = {}) {
  const it = d.item || {};
  const amount = Number.isInteger(d.amount) ? d.amount : null;
  const { duplicate } = await applyStripeEvent(db, event, [unmatchedStmt(db, event, reason, d)]);
  console.warn(JSON.stringify({ msg: 'stripe_payment_unmatched', event_id: event.id, type: event.type, reason, price_id: it.priceId ?? null, amount, duplicate, ...(ALERT_UNMATCHED.has(reason) ? { alert: true } : {}) }));
  return json({ received: true, ignored: true, reason, duplicate }, 200);
}

// LIVE-GATE-1: STRIPE_SECRET_KEY prefix must match LIVE_MODE (live key iff LIVE_MODE
// is the exact string "true"; otherwise a test key). Cheap startsWith. Cloudflare
// Workers have no startup hook, so callers run this at the top of each billing /
// Stripe request, before any Stripe API call or credit change.
export function keyModeGuard(env, headers = {}, extra = {}) {
  if (!env.STRIPE_SECRET_KEY) {
    console.error(JSON.stringify({ msg: 'stripe_secret_key_missing', alert: true, ...extra }));
    return json({ error: 'stripe_secret_key_missing' }, 500, headers);
  }
  if (!stripeKeyMatchesLiveMode(env)) {
    console.error(JSON.stringify({
      msg: 'stripe_key_mode_mismatch',
      alert: true,
      live_mode: isLiveMode(env),
      hint: isLiveMode(env)
        ? 'LIVE_MODE=true requires STRIPE_SECRET_KEY starting with sk_live_ or rk_live_'
        : 'sandbox (LIVE_MODE not the exact string true) requires STRIPE_SECRET_KEY starting with sk_test_ or rk_test_',
      ...extra,
    }));
    return json({ error: 'stripe_key_mode_mismatch' }, 500, headers);
  }
  return null;
}

// Non-2xx answer for a Stripe API failure, so Stripe retries the event later. Nothing is recorded.
// The stripe_webhook_retry line itself is logged once for every non-2xx answer in webhook().
function retryLater(event, error, status, extra = {}) {
  return json({ error, retry: true, ...extra }, status);
}

function nowSeconds() {
  return Math.floor(Date.now() / 1000);
}

// Missing or invalid event.created counts as fresh (Stripe keeps retrying with 502).
function isStripeRetryExhausted(env, event, nowSec = nowSeconds()) {
  const created = event?.created;
  if (!Number.isInteger(created)) return false;
  return (nowSec - created) >= stripeRetryExhaustSeconds(env);
}

// Exhausted transient fetch: one unmatched_payments row, no stripe_events row, 0 credits,
// HTTP 200 so Stripe stops. A later manual resend can still grant. lastStatus goes in
// detail (mode stays the real checkout/invoice mode, or NULL).
async function recordFetchExhausted(db, event, lastStatus, d = {}) {
  const status = lastStatus == null ? 'network' : String(lastStatus);
  await unmatchedStmt(db, event, 'stripe_fetch_exhausted', { ...d, detail: status }).run();
  console.error(JSON.stringify({ msg: 'stripe_fetch_exhausted', alert: true, event_id: event.id, type: event.type, status, created: Number.isInteger(event.created) ? event.created : null }));
  return json({ received: true, ignored: true, reason: 'stripe_fetch_exhausted' }, 200);
}

// GET a Stripe object with the STRIPE_SECRET_KEY Worker secret (prefix must
// already match LIVE_MODE; never committed). Returns { object } or { error, http, ... }.
// STRIPE-404-RETRY-1: any failed fetch (network, timeout, any non-2xx including 404 / 429 / 5xx,
// bad JSON) answers 5xx so Stripe retries. Nothing is recorded (no grant, no claw, no
// stripe_events row, no unmatched_payments) unless the event is older than STRIPE_RETRY_EXHAUST_SECONDS.
//   404 / 429 / 5xx / network / bad JSON -> 502 (or 200 exhausted), log stripe_fetch_transient alert
//   401 / 403                            -> 500 (bad or misconfigured key), log stripe_auth_error alert
// SPROOF-FIX-README-PERMS: restricted-key permission needed for a GET path (hint only; no secrets).
function stripeReadPermissionFor(path) {
  if (path.startsWith('/checkout/sessions')) return 'Checkout Sessions Read';
  if (path.startsWith('/invoice_payments')) return 'Invoice Payments Read';
  if (path.startsWith('/payment_intents')) return 'PaymentIntents Read';
  if (path.startsWith('/invoices')) return 'Invoices Read';
  if (path.startsWith('/charges')) return 'Charges Read';
  if (path.startsWith('/disputes')) return 'Disputes Read';
  return 'read permission';
}

async function fetchStripe(env, event, path) {
  const base = { event_id: event.id, type: event.type };
  // Key/mode already checked at the top of the Stripe request. Missing key is
  // still refused here so a later caller cannot reach the Stripe API.
  if (!env.STRIPE_SECRET_KEY) {
    console.error(JSON.stringify({ msg: 'stripe_secret_key_missing', alert: true, ...base }));
    return { error: 'stripe_secret_key_missing', http: 500 };
  }
  const transient = (status, detail) => {
    console.error(JSON.stringify({ msg: 'stripe_fetch_transient', alert: true, ...base, status, detail }));
    const lastStatus = detail === 'network_error' ? 'network' : detail === 'bad_json' ? 'bad_json' : status;
    return { error: 'stripe_fetch_transient', http: 502, status, lastStatus };
  };
  let r;
  try {
    r = await fetch(`${STRIPE_API}${path}`, { headers: { Authorization: `Bearer ${env.STRIPE_SECRET_KEY}` } });
  } catch (e) { return transient(null, 'network_error'); }
  if (r.status === 401 || r.status === 403) {
    const request = `GET /v1${path.split('?')[0]}`;
    console.error(JSON.stringify({ msg: 'stripe_auth_error', alert: true, ...base, status: r.status, request, hint: `STRIPE_SECRET_KEY is invalid or lacks ${stripeReadPermissionFor(path)} for ${request}` }));
    return { error: 'stripe_auth_error', http: 500, status: r.status };
  }
  if (!r.ok) return transient(r.status, r.status === 429 ? 'rate_limited' : r.status === 404 ? 'not_found' : 'http_error');
  try { return { object: await r.json() }; } catch { return transient(r.status, 'bad_json'); }
}

async function fetchOrRetry(env, event, path, base) {
  const f = await fetchStripe(env, event, path);
  if (f.error === 'stripe_fetch_transient' && isStripeRetryExhausted(env, event)) {
    return { exhausted: await recordFetchExhausted(env.DB, event, f.lastStatus, base) };
  }
  if (f.error) return { retry: retryLater(event, f.error, f.http, f.status != null ? { status: f.status } : {}) };
  return { object: f.object };
}

async function fetchLineItems(env, event, sessionId) {
  if (typeof sessionId !== 'string' || !sessionId) return { notFound: true, reason: 'session_id_missing' };
  const f = await fetchStripe(env, event, `/checkout/sessions/${encodeURIComponent(sessionId)}/line_items?limit=10`);
  if (f.error) return f;
  return { list: f.object };
}

const alreadyProcessed = async (db, event) => !!(await db.prepare('SELECT 1 AS ok FROM stripe_events WHERE event_id = ?1').bind(event.id).first());

// SPROOF-FIX-RENEWAL-IDS: API 2025-03-31.basil removed invoice.payment_intent,
// invoice.charge, and invoice.subscription_details. Metadata lives on
// parent.subscription_details (old subscription_details remains a fallback).
function subscriptionMetadata(o) {
  const parent = o?.parent?.subscription_details?.metadata;
  if (parent && typeof parent === 'object') return parent;
  const legacy = o?.subscription_details?.metadata;
  if (legacy && typeof legacy === 'object') return legacy;
  return undefined;
}

// Paid InvoicePayment row → PaymentIntent id. Same shape on invoice.payments
// and GET /v1/invoice_payments.
function paidPaymentIntentFromPayments(list) {
  const data = list?.data;
  if (!Array.isArray(data)) return null;
  for (const row of data) {
    if (row?.status !== 'paid') continue;
    const pi = stripeId(row?.payment?.payment_intent);
    if (pi) return pi;
  }
  return null;
}

// PI: paid invoice.payments.data[].payment.payment_intent, else the pre-basil
// invoice.payment_intent, else GET /v1/invoice_payments?invoice=<id>.
// Charge: pre-basil invoice.charge, else PaymentIntent.latest_charge via
// fetchOrRetry (transient → 5xx / exhaust window → unmatched 200).
// Missing either id → { missing: true }; caller refuse()s. Never store NULL.
async function resolveRenewalPaymentIds(env, event, o, base) {
  let paymentIntent = paidPaymentIntentFromPayments(o.payments);
  if (!paymentIntent) paymentIntent = stripeId(o.payment_intent);
  if (!paymentIntent) {
    const invoiceId = stripeId(o.id);
    if (invoiceId) {
      const got = await fetchOrRetry(env, event, `/invoice_payments?invoice=${encodeURIComponent(invoiceId)}`, base);
      if (got.exhausted) return got;
      if (got.retry) return got;
      paymentIntent = paidPaymentIntentFromPayments(got.object);
    }
  }

  let charge = stripeId(o.charge);
  if (!charge && paymentIntent) {
    const got = await fetchOrRetry(env, event, `/payment_intents/${encodeURIComponent(paymentIntent)}`, base);
    if (got.exhausted) return got;
    if (got.retry) return got;
    charge = stripeId(got.object?.latest_charge);
  }

  if (!paymentIntent || !charge) return { missing: true };
  return { paymentIntent, charge };
}

// PRICING-4: every non-2xx answer gets one structured stripe_webhook_retry line, so retries are visible.
async function webhook(req, env) {
  const ctx = {};
  const res = await webhookInner(req, env, ctx);
  if (res.status < 200 || res.status > 299) {
    let error = null;
    try { error = (await res.clone().json()).error ?? null; } catch { /* not json */ }
    console.error(JSON.stringify({ msg: 'stripe_webhook_retry', status: res.status, error, event_id: ctx.event?.id ?? null, type: ctx.event?.type ?? null }));
  }
  return res;
}

async function webhookInner(req, env, ctx) {
  const payload = await req.text();
  const ok = await verifyStripeSignature(payload, req.headers.get('Stripe-Signature'), env.STRIPE_WEBHOOK_SECRET);
  if (!ok) return json({ error: 'bad_signature' }, 400);
  let event;
  try { event = JSON.parse(payload); } catch { return json({ error: 'bad_json' }, 400); }
  ctx.event = event;
  const o = event.data?.object || {};
  const gated = keyModeGuard(env, {}, { event_id: event.id, type: event.type });
  if (gated) return gated;
  // Event livemode must match LIVE_MODE. Mismatch goes through refuse() (HTTP 200,
  // unmatched_payments, 0 credits) so Stripe does not retry. Never 400 here.
  const eventLive = event.livemode === true || o.livemode === true;
  if (eventLive !== isLiveMode(env)) {
    console.error(JSON.stringify({ msg: 'livemode_mismatch', alert: true, event_id: event.id, type: event.type, event_livemode: eventLive, live_mode: isLiveMode(env) }));
    return refuse(env.DB, event, 'livemode_mismatch', {
      user: o.client_reference_id || o.metadata?.rathor_user || null,
      amount: Number.isInteger(o.amount_total) ? o.amount_total : (Number.isInteger(o.amount_paid) ? o.amount_paid : null),
      currency: o.currency,
      mode: o.mode ?? o.billing_reason ?? (eventLive ? 'live' : 'test'),
      metadata: o.metadata,
    });
  }
  try {
    return await handleEvent(env, event, o);
  } catch (e) {
    // PRICING-3 / REFUND-1: migrations 0002–0005 not applied. Fail loudly (500) so Stripe retries once fixed.
    if (isSchemaMissing(e)) {
      console.error(JSON.stringify({ msg: 'schema_missing', error: String(e?.message || e), event_id: event.id, type: event.type, hint: 'apply migrations: wrangler d1 migrations apply' }));
      return json({ error: 'schema_missing' }, 500);
    }
    throw e;
  }
}

async function handleEvent(env, event, o) {
  const db = env.DB;

  if (event.type === 'checkout.session.completed') {
    const user = o.client_reference_id || o.metadata?.rathor_user;
    const base = { user: user ?? null, amount: o.amount_total, currency: o.currency, mode: o.mode, metadata: o.metadata };
    if (!user) return refuse(db, event, 'no_user', base);
    if (!(await accountExists(db, user))) return refuse(db, event, 'unknown_user', base);
    // mode=payment => pack/trial prices only; mode=subscription => the monthly plan only.
    const context = o.mode === 'payment' ? 'checkout_payment' : o.mode === 'subscription' ? 'checkout_subscription' : null;
    if (!context) return refuse(db, event, `checkout_mode_${o.mode ?? 'missing'}`, base);
    // A 100%-off code gives payment_status 'no_payment_required': refused, grants nothing.
    if (o.payment_status !== 'paid') return refuse(db, event, `payment_status_${o.payment_status ?? 'missing'}`, base);
    if (Number.isInteger(o.amount_total) && o.amount_total <= 0) return refuse(db, event, 'zero_amount_paid', base);
    if (await alreadyProcessed(db, event)) return json({ received: true, duplicate: true }, 200);
    // PRICING-3: the price id comes from Stripe's API (server-side), never from the payload alone.
    const f = await fetchLineItems(env, event, o.id);
    if (f.error === 'stripe_fetch_transient' && isStripeRetryExhausted(env, event)) {
      return recordFetchExhausted(db, event, f.lastStatus, base);
    }
    if (f.error) return retryLater(event, f.error, f.http, f.status != null ? { status: f.status } : {});
    if (f.notFound) return refuse(db, event, f.reason, base);
    const item = checkoutLineItem(f.list);
    const res = resolveLineItem(env, context, item);
    if (!res.entry) return refuse(db, event, res.reason, { ...base, item });
    return grant(env, event, { user, entry: res.entry, item, reason: 'checkout', customer: o.customer ?? null, email: o.customer_details?.email, metaCredits: o.metadata?.credits, base,
      paymentIntent: stripeId(o.payment_intent), charge: stripeId(o.charge), checkoutSession: stripeId(o.id),
      amountPaid: Number.isInteger(o.amount_total) ? o.amount_total : 0,
      extra: o.subscription ? [db.prepare("UPDATE accounts SET subscription_id = ?2, subscription_status = 'active' WHERE id = ?1").bind(user, o.subscription)] : [] });
  }

  if (event.type === 'invoice.paid') {
    const subId = typeof o.subscription === 'string' ? o.subscription : o.subscription?.id ?? o.parent?.subscription_details?.subscription;
    if (!subId) return json({ received: true, ignored: 'invoice_without_subscription' }, 200);
    // The first invoice (subscription_create) is already granted by checkout.session.completed;
    // only true renewals top up, so month one is not granted twice.
    if (o.billing_reason !== 'subscription_cycle') {
      const { duplicate } = await applyStripeEvent(db, event, []);
      return json({ received: true, duplicate, ignored: `billing_reason_${o.billing_reason ?? 'missing'}` }, 200);
    }
    const acct = await db.prepare('SELECT id, subscription_status FROM accounts WHERE subscription_id = ?1 OR (stripe_customer_id IS NOT NULL AND stripe_customer_id = ?2)').bind(subId, o.customer ?? null).first();
    const subMeta = subscriptionMetadata(o);
    const user = acct?.id || subMeta?.rathor_user || o.metadata?.rathor_user;
    const base = { user: user ?? null, amount: o.amount_paid, currency: o.currency, mode: o.billing_reason, metadata: subMeta };
    // PRICING-4: a renewal paid with a 100%-off coupon (or no amount) grants nothing.
    if (!Number.isInteger(o.amount_paid) || o.amount_paid <= 0) return refuse(db, event, 'zero_amount_paid', base);
    if (!user) return refuse(db, event, 'no_user', base);
    if (!acct && !(await accountExists(db, user))) return refuse(db, event, 'unknown_user', base);
    if (acct && acct.subscription_status && !ACTIVE.has(acct.subscription_status)) {
      await applyStripeEvent(db, event, []);
      return json({ received: true, ignored: 'subscription_not_active' }, 200);
    }
    // Renewals grant only the plan price, checked by the invoice line's price id (amount_paid only
    // has to be > 0; partial coupons and tax change it).
    const item = invoiceLineItem(o);
    const res = resolveLineItem(env, 'invoice_cycle', item);
    if (!res.entry) return refuse(db, event, res.reason, { ...base, item });
    const ids = await resolveRenewalPaymentIds(env, event, o, base);
    if (ids.exhausted) return ids.exhausted;
    if (ids.retry) return ids.retry;
    if (ids.missing) return refuse(db, event, 'renewal_payment_ids_missing', base);
    return grant(env, event, { user, entry: res.entry, item, reason: 'renewal', customer: o.customer ?? null, email: o.customer_email, metaCredits: subMeta?.credits, base,
      paymentIntent: ids.paymentIntent, charge: ids.charge,
      amountPaid: Number.isInteger(o.amount_paid) ? o.amount_paid : 0,
      extra: [db.prepare("UPDATE accounts SET subscription_id = ?2, subscription_status = COALESCE(subscription_status, 'active') WHERE id = ?1").bind(user, subId)] });
  }

  if (event.type === 'customer.subscription.updated' || event.type === 'customer.subscription.deleted') {
    const status = event.type === 'customer.subscription.deleted' ? 'canceled' : String(o.status || 'unknown');
    const stmt = db.prepare('UPDATE accounts SET subscription_status = ?2 WHERE subscription_id = ?1 OR (stripe_customer_id IS NOT NULL AND stripe_customer_id = ?3)').bind(o.id, status, o.customer ?? null);
    const { duplicate } = await applyStripeEvent(db, event, [stmt]);
    return json({ received: true, duplicate, subscription_status: status, renewals: ACTIVE.has(status) ? 'on' : 'stopped' }, 200);
  }

  if (event.type === 'charge.refunded') return handleRefunded(env, event, o);
  if (event.type === 'charge.dispute.created' || event.type === 'charge.dispute.closed') return handleDispute(env, event, o);

  return json({ received: true, ignored: event.type }, 200);
}

async function grant(env, event, g) {
  const db = env.DB;
  const credits = g.entry.credits;
  logMetadataMismatch(event, g.metaCredits, credits);
  const stmts = [...(g.entry.kind === 'trial' ? [trialClaimStmt(db, g.user, event.id)] : []),
    ...grantStmts(db, g.user, credits, g.reason, event.id, {
      customer: g.customer,
      paymentIntent: g.paymentIntent ?? null,
      charge: g.charge ?? null,
      checkoutSession: g.checkoutSession ?? null,
      amountPaid: g.amountPaid ?? 0,
    }),
    // SPROOF-FIX-DISPUTE-ORDER: a dispute that arrived before this grant is applied
    // in the same batch as the credit_grants row (freeze / lost claw / nothing if won).
    ...earlyDisputeGrantStmts(db, g.user, stripeId(g.paymentIntent), stripeId(g.charge), event.id),
    ...g.extra];
  const r = await applyStripeEvent(db, event, stmts);
  if (r.trialUsed) return refuse(db, event, 'trial_already_used', { ...g.base, item: g.item });
  if (r.duplicate) return json({ received: true, duplicate: true }, 200);
  const early = (await db.prepare('SELECT dispute_id, status FROM early_disputes WHERE applied_event_id = ?1').bind(event.id).all()).results || [];
  const emailed = await sendReceipt(env, g.email, credits);
  const out = { received: true, granted: credits, kind: g.entry.kind, matched_by: 'price_id', balance: await available(db, g.user), emailed };
  if (early.length) {
    console.error(JSON.stringify({ msg: 'early_dispute_applied_at_grant', alert: true, event_id: event.id, type: event.type, account_id: g.user, disputes: early }));
    out.early_disputes = early.map((e) => ({ dispute_id: e.dispute_id, status: e.status }));
    out.frozen = await isFrozen(db, g.user);
  }
  return json(out, 200);
}

// REFUND-1 mechanical claw. target = ceil(granted × amount_refunded / amount_paid).
// Only target − already_clawed is removed. Cap at available (balance − open holds)
// so a later settleHold cannot push the balance below 0. One D1 batch CAS:
// UPDATE credit_grants WHERE clawed_credits = already (and available covers actual);
// UPDATE accounts WHERE changes()=1 AND available >= actual; INSERT ledger WHERE
// changes()=1. stripe_events is written only after the CAS wins. One retry on
// conflict; a second loss is 503 claw_conflict with nothing processed.
async function applyClaw(env, event, grant, amountRefunded, { reason = 'refund_claw', freeze = false, freezeReason = 'dispute_lost', extra = [] } = {}) {
  const db = env.DB;
  let row = grant;
  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt > 0) {
      row = await findGrantById(db, grant.id);
      if (!row) return refuse(db, event, 'refund_no_matching_grant', { user: grant.account_id, mode: 'charge' });
    }
    const paid = Number.isInteger(row.amount_paid) ? row.amount_paid : 0;
    const target = clawTarget(row.granted_credits, amountRefunded, paid);
    const already = Number.isInteger(row.clawed_credits) ? row.clawed_credits : 0;
    const delta = Math.max(0, target - already);
    const avail = await available(db, row.account_id);
    const actual = Math.min(delta, Math.max(0, Number(avail) || 0));
    const shortfall = delta - actual;
    const cas = await applyClawCas(db, {
      grantId: row.id, accountId: row.account_id, already, actual, reason, eventId: event.id,
    });
    if (cas.conflict) continue;
    const extras = [...extra];
    if (shortfall > 0) {
      extras.push(freezeStmt(db, row.account_id, 'refund_claw_shortfall'));
      extras.push(unmatchedStmt(db, event, 'refund_claw_shortfall', {
        user: row.account_id, amount: amountRefunded, mode: 'charge', detail: shortfall,
      }));
      console.error(JSON.stringify({
        msg: 'refund_claw_shortfall', alert: true, event_id: event.id, type: event.type,
        account_id: row.account_id, grant_id: row.id, target, already, delta, actual, shortfall, available: avail,
      }));
    } else if (freeze) {
      extras.push(freezeStmt(db, row.account_id, freezeReason));
    }
    const r = await applyStripeEvent(db, event, extras);
    if (r.duplicate) return json({ received: true, duplicate: true }, 200);
    return json({
      received: true,
      clawed: actual,
      target,
      already_clawed: already + actual,
      shortfall,
      frozen: shortfall > 0 || freeze,
      balance: await available(db, row.account_id),
    }, 200);
  }
  console.error(JSON.stringify({ msg: 'claw_conflict', alert: true, event_id: event.id, type: event.type, grant_id: grant.id }));
  return json({ error: 'claw_conflict', retry: true }, 503);
}

async function handleRefunded(env, event, o) {
  const db = env.DB;
  if (await alreadyProcessed(db, event)) return json({ received: true, duplicate: true }, 200);
  const chargeId = stripeId(o.id);
  const base = {
    user: null,
    amount: Number.isInteger(o.amount_refunded) ? o.amount_refunded : (Number.isInteger(o.amount) ? o.amount : null),
    currency: o.currency,
    mode: 'charge',
  };
  if (!chargeId) return refuse(db, event, 'refund_no_matching_grant', { ...base, detail: 'charge_id_missing' });
  const got = await fetchOrRetry(env, event, `/charges/${encodeURIComponent(chargeId)}`, base);
  if (got.exhausted) return got.exhausted;
  if (got.retry) return got.retry;
  const charge = got.object;
  const grant = await findGrant(db, { paymentIntent: charge.payment_intent, charge: charge.id });
  if (!grant) return refuse(db, event, 'refund_no_matching_grant', {
    ...base,
    user: await findAccountByCustomer(db, charge.customer),
    amount: Number.isInteger(charge.amount_refunded) ? charge.amount_refunded : base.amount,
    currency: charge.currency ?? base.currency,
    detail: stripeId(charge.id) || chargeId,
  });
  if (!grant.charge_id && stripeId(charge.id)) {
    await db.prepare('UPDATE credit_grants SET charge_id = ?2 WHERE id = ?1 AND charge_id IS NULL').bind(grant.id, stripeId(charge.id)).run();
    grant.charge_id = stripeId(charge.id);
  }
  const refunded = Number.isInteger(charge.amount_refunded) ? charge.amount_refunded : 0;
  return applyClaw(env, event, grant, refunded);
}

async function handleDispute(env, event, o) {
  const db = env.DB;
  if (await alreadyProcessed(db, event)) return json({ received: true, duplicate: true }, 200);
  const disputeId = stripeId(o.id);
  const base = {
    user: null,
    amount: Number.isInteger(o.amount) ? o.amount : null,
    currency: o.currency,
    mode: 'dispute',
  };
  if (!disputeId) return refuse(db, event, 'dispute_no_matching_grant', { ...base, detail: 'dispute_id_missing' });
  const got = await fetchOrRetry(env, event, `/disputes/${encodeURIComponent(disputeId)}`, base);
  if (got.exhausted) return got.exhausted;
  if (got.retry) return got.retry;
  const d = got.object;
  const status = String(d.status || 'unknown');
  const chargeId = stripeId(d.charge);
  const pi = stripeId(d.payment_intent);
  const grant = await findGrant(db, { paymentIntent: d.payment_intent, charge: d.charge });
  // SPROOF-FIX-DISPUTE-ORDER: no grant yet (dispute before invoice.paid / checkout).
  // Store it durably keyed by PI and charge, so grant() applies it in its own batch.
  const early = grant ? null : earlyDisputeUpsertStmt(db, {
    disputeId, status, paymentIntent: pi, charge: chargeId,
    amount: Number.isInteger(d.amount) ? d.amount : base.amount, currency: d.currency ?? base.currency, eventId: event.id,
  });
  // SPROOF-FIX-DISPUTE-RACE: whenever the grant is missing (account found by customer
  // or not), store it and, while fresh, answer a transient 5xx so Stripe retries (same
  // exhaust clock as fetchOrRetry). A lost dispute is then clawed by grant() or by a
  // retry that finds the grant. After the window, fall through to the existing paths.
  if (!grant) {
    await early.run();
    if (!isStripeRetryExhausted(env, event)) {
      console.error(JSON.stringify({ msg: 'dispute_stored_early', alert: true, event_id: event.id, type: event.type, dispute_id: disputeId, status, payment_intent: pi, charge: chargeId }));
      return retryLater(event, 'dispute_grant_pending', 503, { dispute_id: disputeId, stored: true });
    }
  }
  const user = grant?.account_id || await findAccountByCustomer(db, d.customer);
  if (!user) {
    return refuse(db, event, 'dispute_no_matching_grant', {
      ...base,
      amount: Number.isInteger(d.amount) ? d.amount : base.amount,
      currency: d.currency ?? base.currency,
      detail: disputeId,
    });
  }
  if (event.type === 'charge.dispute.created') {
    const existing = await db.prepare('SELECT status FROM stripe_disputes WHERE dispute_id = ?1').bind(disputeId).first();
    // A won/warning_closed/lost row arriving before created must not later refreeze.
    if (existing && isDisputeTerminalStatus(existing.status)) {
      const { duplicate } = await applyStripeEvent(db, event, early ? [early] : []);
      if (duplicate) return json({ received: true, duplicate: true }, 200);
      return json({ received: true, frozen: await isFrozen(db, user), dispute_id: disputeId, ignored: 'dispute_already_closed' }, 200);
    }
    const { duplicate } = await applyStripeEvent(db, event, [
      freezeStmt(db, user, 'dispute'),
      disputeUpsertStmt(db, disputeId, user, status, chargeId, pi, event.id),
      ...(early ? [early] : []),
    ]);
    if (duplicate) return json({ received: true, duplicate: true }, 200);
    return json({ received: true, frozen: true, dispute_id: disputeId }, 200);
  }
  // charge.dispute.closed
  if (isDisputeUnfreezeStatus(status)) {
    const { duplicate } = await applyStripeEvent(db, event, [
      disputeUpsertStmt(db, disputeId, user, status, chargeId, pi, event.id),
      unfreezeIfNoOpenDisputesStmt(db, user),
      ...(early ? [early] : []),
    ]);
    if (duplicate) return json({ received: true, duplicate: true }, 200);
    return json({ received: true, frozen: await isFrozen(db, user), dispute_status: status }, 200);
  }
  // lost (and any other closed status): claw like a full refund, stay frozen
  // until a human clears accounts.frozen. Documented in README.
  if (grant) {
    const paid = Number.isInteger(grant.amount_paid) && grant.amount_paid > 0 ? grant.amount_paid : (Number.isInteger(d.amount) ? d.amount : 0);
    return applyClaw(env, event, grant, paid, {
      reason: 'dispute_lost_claw',
      freeze: true,
      freezeReason: 'dispute_lost',
      extra: [disputeUpsertStmt(db, disputeId, user, status, chargeId, pi, event.id)],
    });
  }
  const { duplicate } = await applyStripeEvent(db, event, [
    freezeStmt(db, user, 'dispute_lost'),
    disputeUpsertStmt(db, disputeId, user, status, chargeId, pi, event.id),
    unmatchedStmt(db, event, 'dispute_no_matching_grant', { ...base, user, detail: disputeId }),
    early,
  ]);
  if (duplicate) return json({ received: true, duplicate: true }, 200);
  console.error(JSON.stringify({ msg: 'dispute_no_matching_grant', alert: true, event_id: event.id, type: event.type, dispute_id: disputeId, account_id: user }));
  return json({ received: true, frozen: true, dispute_status: status, clawed: 0 }, 200);
}

// GET /v1/credits — balance read. Frozen accounts may still read.
async function readCredits(req, env, h) {
  const allowed = env.ALLOWED_ORIGIN || 'https://rathor.ai';
  const byKey = await authenticateBearer(req, env);
  let user;
  if (byKey) {
    if (req.headers.get('Origin') && req.headers.get('Origin') !== allowed) return json({ error: 'origin_not_allowed' }, 403, h);
    user = byKey;
  } else if (parseCookie(req)) {
    user = await authenticateSession(req, env);
    if (!user) return json({ error: 'unauthorized' }, 401, h);
  } else {
    return json({ error: 'unauthorized' }, 401, h);
  }
  const row = await accountBalance(env.DB, user);
  return json({
    credits: await available(env.DB, user),
    balance: row.balance ?? 0,
    frozen: !!(row.frozen || row.settle_locked),
  }, 200, { ...h, 'Cache-Control': 'no-store' });
}

// WEB-AUTH-1: signed-in checkout. Price key → STRIPE_PRICE_* var. Never trust
// client-sent amounts, credits, or Stripe price ids. Returns only { url }.
async function checkout(req, env, h) {
  const gate = await requireSessionCsrf(req, env, h);
  if (gate.response) return gate.response;
  if (await isFrozen(env.DB, gate.user)) return json({ error: 'account_frozen' }, 403, h);
  const gated = keyModeGuard(env, h);
  if (gated) return gated;
  let body;
  try { body = await req.json(); } catch { return json({ error: 'bad_json' }, 400, h); }
  const mapped = resolveCheckoutPrice(env, body?.price);
  if (mapped.reason === 'unknown_price_key') return json({ error: 'unknown_price_key' }, 400, h);
  if (mapped.reason === 'price_not_configured') return json({ error: 'price_not_configured' }, 503, h);
  if (mapped.spec.trial && await trialAlreadyUsed(env.DB, gate.user)) {
    return json({ error: 'trial_already_used' }, 409, h);
  }
  const allowed = env.ALLOWED_ORIGIN || 'https://rathor.ai';
  const success = env.STRIPE_CHECKOUT_SUCCESS_URL || `${allowed}/?checkout=success`;
  const cancel = env.STRIPE_CHECKOUT_CANCEL_URL || `${allowed}/?checkout=cancel`;
  const form = new URLSearchParams();
  form.set('mode', mapped.spec.mode);
  form.set('client_reference_id', gate.user);
  form.set('success_url', success);
  form.set('cancel_url', cancel);
  form.set('line_items[0][price]', mapped.priceId);
  form.set('line_items[0][quantity]', '1');
  form.set('metadata[rathor_user]', gate.user);
  let r;
  try {
    r = await fetch(`${STRIPE_API}/checkout/sessions`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${env.STRIPE_SECRET_KEY}`, 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form,
    });
  } catch { return json({ error: 'stripe_checkout_failed' }, 502, h); }
  if (!r.ok) {
    if (r.status >= 400 && r.status < 500) {
      // Stripe error code/type only; never the key or request body.
      let code = null; let type = null;
      try { const e = (await r.json())?.error; code = e?.code ?? null; type = e?.type ?? null; } catch { /* not json */ }
      console.error(JSON.stringify({ msg: 'stripe_checkout_4xx', status: r.status, stripe_code: code, stripe_type: type, request: 'POST /v1/checkout/sessions', ...(r.status === 401 || r.status === 403 ? { hint: 'STRIPE_SECRET_KEY is invalid or lacks Checkout Sessions Write' } : {}) }));
    }
    return json({ error: 'stripe_checkout_failed' }, 502, h);
  }
  let data;
  try { data = await r.json(); } catch { return json({ error: 'stripe_checkout_failed' }, 502, h); }
  if (typeof data.url !== 'string' || !data.url) return json({ error: 'stripe_checkout_failed' }, 502, h);
  return json({ url: data.url }, 200, h);
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const h = cors(env, req);
    if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: h });
    if (req.method === 'GET' && url.pathname === '/health') return json({ ok: true, service: 'rathor-api', mode: 'sandbox', rates_verified: ratesVerified(env) }, 200, h);
    if (req.method === 'POST' && url.pathname === '/v1/auth/request') return requestMagicLink(req, env, h);
    if (req.method === 'GET' && url.pathname === '/v1/auth/verify') return verifyConfirmPage(req, env, h);
    if (req.method === 'POST' && url.pathname === '/v1/auth/verify') return verifyMagicLink(req, env, h);
    if (req.method === 'POST' && url.pathname === '/v1/auth/logout') return logout(req, env, h);
    if (req.method === 'POST' && url.pathname === '/v1/checkout') return checkout(req, env, h);
    if (req.method === 'POST' && url.pathname === '/v1/keys') return issueKey(req, env, h);
    if (req.method === 'POST' && url.pathname === '/v1/keys/revoke') return revokeKeys(req, env, h);
    if (req.method === 'GET' && url.pathname === '/v1/credits') return readCredits(req, env, h);
    if (req.method === 'POST' && url.pathname === '/v1/chat') return chat(req, env, h);
    if (req.method === 'POST' && url.pathname === '/v1/stripe/webhook') return webhook(req, env);
    return json({ error: 'not_found' }, 404, h);
  },
};
