import { describe, it, expect, vi, beforeEach } from 'vitest';
import worker from '../src/index.js';
import { hmacHex } from '../src/stripe.js';
import { sha256Hex } from '../src/auth.js';
import { clawTarget, stripeId, placeHold, settleHold, releaseHold, unlockSettleOverHold, pendingEarlyDisputes } from '../src/db.js';
import { stripeRetryExhaustSeconds, STRIPE_RETRY_EXHAUST_SECONDS_DEFAULT, STRIPE_RETRY_EXHAUST_SECONDS_MIN, SESSION_COOKIE_NAME, CSRF_HEADER } from '../src/config.js';
import { fakeD1 } from './fakeD1.js';

const UNIT = { price_test_p10: 1000, price_test_p25: 2500, price_test_p50: 5000, price_test_plan: 1000, price_test_trial: 100 };
const li = (id, { qty = 1, unit = UNIT[id] ?? 1000, currency = 'usd', subtotal, discount = 0, tax = 0 } = {}) => {
  const sub = subtotal ?? unit * qty;
  return { object: 'list', has_more: false, data: [{ id: 'li_test', object: 'item', quantity: qty, currency, amount_subtotal: sub, amount_discount: discount, amount_tax: tax, amount_total: sub - discount + tax, price: { id, object: 'price', unit_amount: unit, currency } }] };
};

const O = 'https://rathor.ai';
const PRICES = { STRIPE_PRICE_PACK_10: 'price_test_p10', STRIPE_PRICE_PACK_25: 'price_test_p25', STRIPE_PRICE_PACK_50: 'price_test_p50', STRIPE_PRICE_PLAN_MONTHLY: 'price_test_plan', STRIPE_PRICE_TRIAL: 'price_test_trial' };
const KEY = ['rth', 'test', '0123456789abcdefghijklmnop'].join('_');
const STRIPE_API_OBJECTS = new Map();
let env;

beforeEach(async () => {
  vi.restoreAllMocks();
  STRIPE_API_OBJECTS.clear();
  env = { ALLOWED_ORIGIN: O, DB: fakeD1(), XAI_API_KEY: 'test', STRIPE_WEBHOOK_SECRET: 'whsec_test', RESEND_API_KEY: 'test', RATES_VERIFIED: 'true', STRIPE_SECRET_KEY: 'sk_test_FAKE_not_a_real_key', ...PRICES };
  env.DB.q("INSERT INTO accounts (id, balance) VALUES ('u1', 1000)");
  env.DB.q("INSERT INTO ledger (account_id, delta, reason) VALUES ('u1', 1000, 'seed')");
  env.DB.q('INSERT INTO api_keys (key_hash, account_id) VALUES (?1, ?2)', await sha256Hex(KEY), 'u1');
});

function assertLedgerMatchesBalance(id = 'u1') {
  const row = env.DB.q(
    `SELECT a.balance AS balance,
            COALESCE((SELECT SUM(delta) FROM ledger l WHERE l.account_id = a.id), 0) AS led
     FROM accounts a WHERE a.id = ?1`,
    id,
  )[0];
  expect(row.led).toBe(row.balance);
}

const bal = (id = 'u1') => env.DB.q('SELECT balance FROM accounts WHERE id = ?1', id)[0]?.balance;
const frozen = (id = 'u1') => env.DB.q('SELECT frozen, freeze_reason FROM accounts WHERE id = ?1', id)[0];
const unmatched = () => env.DB.q('SELECT * FROM unmatched_payments');
const grants = () => env.DB.q('SELECT * FROM credit_grants');

async function signed(evt, secret = 'whsec_test') {
  const ob = evt.data?.object;
  if (evt.type === 'checkout.session.completed' && ob && 'line_items' in ob) {
    evt = structuredClone(evt);
    STRIPE_API_OBJECTS.set(`li:${evt.data.object.id}`, evt.data.object.line_items);
    delete evt.data.object.line_items;
  }
  const body = JSON.stringify(evt);
  const t = Math.floor(Date.now() / 1000);
  return new Request('https://api.rathor.ai/v1/stripe/webhook', { method: 'POST', body, headers: { 'Stripe-Signature': `t=${t},v1=${await hmacHex(secret, `${t}.${body}`)}` } });
}

const checkoutEvt = (o = {}) => {
  const e = {
    id: 'evt_g1',
    type: 'checkout.session.completed',
    livemode: false,
    data: {
      object: {
        line_items: li('price_test_p10'),
        client_reference_id: 'u1',
        mode: 'payment',
        payment_status: 'paid',
        amount_total: 1000,
        currency: 'usd',
        payment_intent: 'pi_ref_1',
        customer: 'cus_1',
        metadata: {},
        customer_details: { email: 'test@example.com' },
      },
    },
    ...o,
  };
  e.data.object.id ??= `cs_test_${e.id}`;
  return e;
};

const stripeFetch = (onMatch) => vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
  const u = String(url);
  if (onMatch) {
    const override = await onMatch(u, init);
    if (override) return override;
  }
  const liM = u.match(/^https:\/\/api\.stripe\.com\/v1\/checkout\/sessions\/([^/]+)\/line_items/);
  if (liM) {
    const l = STRIPE_API_OBJECTS.get(`li:${decodeURIComponent(liM[1])}`);
    return l ? new Response(JSON.stringify(l)) : new Response('{"error":{"type":"invalid_request_error"}}', { status: 404 });
  }
  const ch = u.match(/^https:\/\/api\.stripe\.com\/v1\/charges\/([^/?]+)/);
  if (ch) {
    const obj = STRIPE_API_OBJECTS.get(`ch:${decodeURIComponent(ch[1])}`);
    return obj ? new Response(JSON.stringify(obj)) : new Response('{"error":{"type":"invalid_request_error"}}', { status: 404 });
  }
  const dp = u.match(/^https:\/\/api\.stripe\.com\/v1\/disputes\/([^/?]+)/);
  if (dp) {
    const obj = STRIPE_API_OBJECTS.get(`dp:${decodeURIComponent(dp[1])}`);
    return obj ? new Response(JSON.stringify(obj)) : new Response('{"error":{"type":"invalid_request_error"}}', { status: 404 });
  }
  return new Response('{}');
});

function putCharge(id, fields) {
  const obj = { id, object: 'charge', currency: 'usd', customer: 'cus_1', ...fields };
  STRIPE_API_OBJECTS.set(`ch:${id}`, obj);
  return obj;
}

function putDispute(id, fields) {
  const obj = { id, object: 'dispute', currency: 'usd', ...fields };
  STRIPE_API_OBJECTS.set(`dp:${id}`, obj);
  return obj;
}

function refundEvt(id, charge, extras = {}) {
  return { id, type: 'charge.refunded', livemode: false, data: { object: { ...charge, ...extras } } };
}

function disputeEvt(id, type, dispute, extras = {}) {
  return { id, type, livemode: false, data: { object: { ...dispute, ...extras } } };
}

const chatReq = (body = { messages: [{ role: 'user', content: 'hi' }] }) =>
  new Request('https://api.rathor.ai/v1/chat', { method: 'POST', body: JSON.stringify(body), headers: { Origin: O, Authorization: `Bearer ${KEY}` } });

const creditsReq = () =>
  new Request('https://api.rathor.ai/v1/credits', { method: 'GET', headers: { Origin: O, Authorization: `Bearer ${KEY}` } });

async function grantPack({ id = 'evt_g1', pi = 'pi_ref_1', amount = 1000 } = {}) {
  stripeFetch();
  const e = checkoutEvt({ id });
  e.data.object.payment_intent = pi;
  e.data.object.amount_total = amount;
  const j = await (await worker.fetch(await signed(e), env)).json();
  expect(j.granted).toBe(10000);
  return { pi, session: e.data.object.id };
}

describe('REFUND-1 clawTarget', () => {
  it('full refund claws the whole grant; partial rounds up', () => {
    expect(clawTarget(10000, 1000, 1000)).toBe(10000);
    expect(clawTarget(10000, 1000, 3000)).toBe(3334); // ceil(10000 * 1000 / 3000)
    expect(clawTarget(10000, 1, 1000)).toBe(10);
    expect(clawTarget(10000, 0, 1000)).toBe(0);
    expect(clawTarget(0, 1000, 1000)).toBe(0);
    expect(stripeId('pi_1')).toBe('pi_1');
    expect(stripeId({ id: 'ch_1' })).toBe('ch_1');
    expect(stripeId(null)).toBeNull();
  });
});

describe('REFUND-1 exhaust floor', () => {
  it('values under 3600 including 0 fall back to 262800', () => {
    expect(STRIPE_RETRY_EXHAUST_SECONDS_MIN).toBe(3600);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: '0' })).toBe(STRIPE_RETRY_EXHAUST_SECONDS_DEFAULT);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: '1' })).toBe(262800);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: '3599' })).toBe(262800);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: '3600' })).toBe(3600);
  });
});

describe('REFUND-1 migration 0005', () => {
  it('adds detail column and unique (stripe_event_id, reason)', () => {
    env.DB.q("INSERT INTO unmatched_payments (stripe_event_id, event_type, reason, detail) VALUES ('e1', 't', 'r', 'hello')");
    expect(env.DB.q("SELECT detail FROM unmatched_payments WHERE stripe_event_id='e1'")[0].detail).toBe('hello');
    expect(() => env.DB.q("INSERT INTO unmatched_payments (stripe_event_id, event_type, reason, detail) VALUES ('e1', 't', 'r', 'dup')")).toThrow(/UNIQUE/);
    env.DB.q("INSERT INTO unmatched_payments (stripe_event_id, event_type, reason, detail) VALUES ('e1', 't', 'other', 'ok')");
    env.DB.q("INSERT OR IGNORE INTO unmatched_payments (stripe_event_id, event_type, reason, detail) VALUES ('e1', 't', 'r', 'ignored')");
    expect(env.DB.q("SELECT COUNT(*) AS n FROM unmatched_payments WHERE stripe_event_id='e1' AND reason='r'")[0].n).toBe(1);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM unmatched_payments WHERE mode = '404'")[0].n).toBe(0);
  });
});

describe('REFUND-1 charge.refunded', () => {
  it('full refund claws the credits that grant added', async () => {
    await grantPack();
    expect(bal()).toBe(11000);
    const charge = putCharge('ch_full', { amount: 1000, amount_refunded: 1000, payment_intent: 'pi_ref_1' });
    const j = await (await worker.fetch(await signed(refundEvt('evt_rf_full', charge)), env)).json();
    expect(j.clawed).toBe(10000);
    expect(j.shortfall).toBe(0);
    expect(bal()).toBe(1000);
    expect(bal()).toBeGreaterThanOrEqual(0);
    expect(grants()[0].clawed_credits).toBe(10000);
    expect(env.DB.q("SELECT delta, reason FROM ledger WHERE stripe_event_id='evt_rf_full'")[0]).toMatchObject({ delta: -10000, reason: 'refund_claw' });
    assertLedgerMatchesBalance();
  });

  it('partial refund rounds the claw target up', async () => {
    await grantPack({ amount: 3000 });
    const charge = putCharge('ch_part', { amount: 3000, amount_refunded: 1000, payment_intent: 'pi_ref_1' });
    const j = await (await worker.fetch(await signed(refundEvt('evt_rf_part', charge)), env)).json();
    expect(j.target).toBe(3334);
    expect(j.clawed).toBe(3334);
    expect(bal()).toBe(11000 - 3334);
    expect(grants()[0].clawed_credits).toBe(3334);
    assertLedgerMatchesBalance();
  });

  it('repeated charge.refunded with cumulative amount_refunded claws only the delta', async () => {
    await grantPack({ amount: 3000 });
    const first = putCharge('ch_cum', { amount: 3000, amount_refunded: 1000, payment_intent: 'pi_ref_1' });
    expect((await (await worker.fetch(await signed(refundEvt('evt_rf_c1', first)), env)).json()).clawed).toBe(3334);
    expect(bal()).toBe(11000 - 3334);
    const second = putCharge('ch_cum', { amount: 3000, amount_refunded: 2000, payment_intent: 'pi_ref_1' });
    const j = await (await worker.fetch(await signed(refundEvt('evt_rf_c2', second)), env)).json();
    expect(j.target).toBe(6667); // ceil(10000 * 2000 / 3000)
    expect(j.clawed).toBe(6667 - 3334);
    expect(bal()).toBe(11000 - 6667);
    expect(grants()[0].clawed_credits).toBe(6667);
    const again = await (await worker.fetch(await signed(refundEvt('evt_rf_c2', second)), env)).json();
    expect(again.duplicate).toBe(true);
    expect(bal()).toBe(11000 - 6667);
    const sameAmt = await (await worker.fetch(await signed(refundEvt('evt_rf_c3', second)), env)).json();
    expect(sameAmt.clawed).toBe(0);
    expect(bal()).toBe(11000 - 6667);
    assertLedgerMatchesBalance();
  });

  it('refund with no matching grant writes one unmatched row and returns 200', async () => {
    const err = vi.spyOn(console, 'warn').mockImplementation(() => {});
    stripeFetch();
    const charge = putCharge('ch_none', { amount: 1000, amount_refunded: 1000, payment_intent: 'pi_unknown' });
    const r = await worker.fetch(await signed(refundEvt('evt_rf_none', charge)), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'refund_no_matching_grant' });
    expect(bal()).toBe(1000);
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_rf_none', reason: 'refund_no_matching_grant' }]);
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).find((x) => x.reason === 'refund_no_matching_grant')).toMatchObject({ alert: true });
  });

  it('shortfall claws down to 0, freezes, alerts, and writes one unmatched row; balance never goes negative', async () => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    await grantPack();
    env.DB.q("UPDATE accounts SET balance = 500 WHERE id='u1'");
    env.DB.q("INSERT INTO ledger (account_id, delta, reason) VALUES ('u1', -10500, 'chat')");
    const charge = putCharge('ch_short', { amount: 1000, amount_refunded: 1000, payment_intent: 'pi_ref_1' });
    const r = await worker.fetch(await signed(refundEvt('evt_rf_short', charge)), env);
    expect(r.status).toBe(200);
    const j = await r.json();
    expect(j.clawed).toBe(500);
    expect(j.shortfall).toBe(9500);
    expect(j.frozen).toBe(true);
    expect(bal()).toBe(0);
    expect(bal()).toBeGreaterThanOrEqual(0);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'refund_claw_shortfall' });
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_rf_short', reason: 'refund_claw_shortfall', detail: '9500', account_id: 'u1' }]);
    expect(grants()[0].clawed_credits).toBe(500);
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).find((x) => x.msg === 'refund_claw_shortfall')).toMatchObject({ alert: true, shortfall: 9500 });
    assertLedgerMatchesBalance();
  });

  it('livemode mismatch on charge.refunded goes to refuse() and does not fetch', async () => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const f = stripeFetch();
    const charge = putCharge('ch_live', { amount: 1000, amount_refunded: 1000, payment_intent: 'pi_ref_1' });
    const r = await worker.fetch(await signed(refundEvt('evt_rf_live', charge, { livemode: true })), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'livemode_mismatch' });
    expect(bal()).toBe(1000);
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_rf_live', reason: 'livemode_mismatch' }]);
    expect(f.mock.calls.filter((c) => String(c[0]).includes('/charges/'))).toHaveLength(0);
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).find((x) => x.msg === 'livemode_mismatch')).toMatchObject({ alert: true });
  });

  it('Stripe fetch failure on charge.refunded is 502 with nothing written', async () => {
    await grantPack();
    const r = await worker.fetch(await signed(refundEvt('evt_rf_502', { id: 'ch_missing', amount: 1000, amount_refunded: 1000, payment_intent: 'pi_ref_1' })), env);
    expect(r.status).toBe(502);
    expect(await r.json()).toMatchObject({ error: 'stripe_fetch_transient', retry: true });
    expect(bal()).toBe(11000);
    expect(unmatched()).toHaveLength(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_rf_502'")[0].n).toBe(0);
    expect(grants()[0].clawed_credits).toBe(0);
  });
});

describe('REFUND-1 charge.dispute', () => {
  it('dispute.created freezes chat (403) while the balance read still works', async () => {
    await grantPack();
    const d = putDispute('dp_1', { status: 'needs_response', charge: 'ch_d1', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    const j = await (await worker.fetch(await signed(disputeEvt('evt_dp_c', 'charge.dispute.created', d)), env)).json();
    expect(j.frozen).toBe(true);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute' });
    const chat = await worker.fetch(chatReq(), env);
    expect(chat.status).toBe(403);
    expect(await chat.json()).toMatchObject({ error: 'account_frozen' });
    const read = await worker.fetch(creditsReq(), env);
    expect(read.status).toBe(200);
    expect(read.headers.get('Cache-Control')).toBe('no-store');
    expect(await read.json()).toMatchObject({ credits: 11000, balance: 11000, frozen: true });
    assertLedgerMatchesBalance();
  });

  it('dispute.closed won unfreezes spending', async () => {
    await grantPack();
    const open = putDispute('dp_won', { status: 'needs_response', charge: 'ch_dw', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    await worker.fetch(await signed(disputeEvt('evt_dp_w1', 'charge.dispute.created', open)), env);
    const closed = putDispute('dp_won', { status: 'won', charge: 'ch_dw', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    const j = await (await worker.fetch(await signed(disputeEvt('evt_dp_w2', 'charge.dispute.closed', closed)), env)).json();
    expect(j.frozen).toBe(false);
    expect(j.dispute_status).toBe('won');
    expect(frozen().frozen).toBe(0);
    vi.mocked(globalThis.fetch).mockRestore?.();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }], usage: { prompt_tokens: 1, completion_tokens: 1 } })));
    expect((await worker.fetch(chatReq(), env)).status).toBe(200);
    assertLedgerMatchesBalance();
  });

  it('dispute.closed lost claws like a full refund and stays frozen', async () => {
    await grantPack();
    const open = putDispute('dp_lost', { status: 'needs_response', charge: 'ch_dl', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    await worker.fetch(await signed(disputeEvt('evt_dp_l1', 'charge.dispute.created', open)), env);
    const closed = putDispute('dp_lost', { status: 'lost', charge: 'ch_dl', payment_intent: 'pi_ref_1', amount: 1000, customer: 'cus_1' });
    const j = await (await worker.fetch(await signed(disputeEvt('evt_dp_l2', 'charge.dispute.closed', closed)), env)).json();
    expect(j.clawed).toBe(10000);
    expect(j.frozen).toBe(true);
    expect(bal()).toBe(1000);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute_lost' });
    expect((await worker.fetch(chatReq(), env)).status).toBe(403);
    const read = await worker.fetch(creditsReq(), env);
    expect(read.status).toBe(200);
    expect(await read.json()).toMatchObject({ frozen: true, balance: 1000 });
    assertLedgerMatchesBalance();
  });

  it('livemode mismatch on charge.dispute.created goes to refuse()', async () => {
    stripeFetch();
    const d = putDispute('dp_live', { status: 'needs_response', charge: 'ch_x', payment_intent: 'pi_x' });
    const r = await worker.fetch(await signed(disputeEvt('evt_dp_live', 'charge.dispute.created', d, { livemode: true })), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'livemode_mismatch' });
    expect(frozen().frozen).toBe(0);
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_dp_live', reason: 'livemode_mismatch' }]);
  });

  it('Stripe fetch failure on charge.dispute.created is 502 with nothing written', async () => {
    await grantPack();
    const r = await worker.fetch(await signed(disputeEvt('evt_dp_502', 'charge.dispute.created', { id: 'dp_missing', status: 'needs_response' })), env);
    expect(r.status).toBe(502);
    expect(await r.json()).toMatchObject({ error: 'stripe_fetch_transient', retry: true });
    expect(frozen().frozen).toBe(0);
    expect(unmatched()).toHaveLength(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_dp_502'")[0].n).toBe(0);
  });

  it('closing one of two disputes as won keeps the account frozen; closing both unfreezes', async () => {
    await grantPack();
    const a = putDispute('dp_a', { status: 'needs_response', charge: 'ch_a', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    const b = putDispute('dp_b', { status: 'needs_response', charge: 'ch_b', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    await worker.fetch(await signed(disputeEvt('evt_dp_a1', 'charge.dispute.created', a)), env);
    await worker.fetch(await signed(disputeEvt('evt_dp_b1', 'charge.dispute.created', b)), env);
    expect(frozen().frozen).toBe(1);
    putDispute('dp_a', { status: 'won', charge: 'ch_a', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    const one = await (await worker.fetch(await signed(disputeEvt('evt_dp_a2', 'charge.dispute.closed', STRIPE_API_OBJECTS.get('dp:dp_a'))), env)).json();
    expect(one.frozen).toBe(true);
    expect(frozen().frozen).toBe(1);
    putDispute('dp_b', { status: 'won', charge: 'ch_b', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    const two = await (await worker.fetch(await signed(disputeEvt('evt_dp_b2', 'charge.dispute.closed', STRIPE_API_OBJECTS.get('dp:dp_b'))), env)).json();
    expect(two.frozen).toBe(false);
    expect(frozen().frozen).toBe(0);
    assertLedgerMatchesBalance();
  });

  it('a won event before created does not later refreeze on a late created', async () => {
    await grantPack();
    const closed = putDispute('dp_late', { status: 'won', charge: 'ch_late', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    const w = await (await worker.fetch(await signed(disputeEvt('evt_dp_late_w', 'charge.dispute.closed', closed)), env)).json();
    expect(w.frozen).toBe(false);
    expect(frozen().frozen).toBe(0);
    putDispute('dp_late', { status: 'needs_response', charge: 'ch_late', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    const c = await (await worker.fetch(await signed(disputeEvt('evt_dp_late_c', 'charge.dispute.created', STRIPE_API_OBJECTS.get('dp:dp_late'))), env)).json();
    expect(c.ignored).toBe('dispute_already_closed');
    expect(c.frozen).toBe(false);
    expect(frozen().frozen).toBe(0);
    expect(env.DB.q("SELECT status FROM stripe_disputes WHERE dispute_id='dp_late'")[0].status).toBe('won');
    assertLedgerMatchesBalance();
  });
});

describe('REFUND-1 atomic claw and freeze surfaces', () => {
  it('two concurrent refunds on the same grant claw the correct total exactly once', async () => {
    await grantPack();
    const charge = putCharge('ch_race', { amount: 1000, amount_refunded: 1000, payment_intent: 'pi_ref_1' });
    const [r1, r2] = await Promise.all([
      worker.fetch(await signed(refundEvt('evt_race_1', charge)), env),
      worker.fetch(await signed(refundEvt('evt_race_2', charge)), env),
    ]);
    expect(r1.status).toBe(200);
    expect(r2.status).toBe(200);
    const j1 = await r1.json();
    const j2 = await r2.json();
    expect([j1.clawed, j2.clawed].sort((a, b) => a - b)).toEqual([0, 10000]);
    expect(grants()[0].clawed_credits).toBe(10000);
    expect(bal()).toBe(1000);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE reason='refund_claw' AND account_id='u1'")[0].n).toBe(1);
    assertLedgerMatchesBalance();
  });

  it('a lost dispute and a refund at the same time claw the grant once', async () => {
    await grantPack();
    const charge = putCharge('ch_both', { amount: 1000, amount_refunded: 1000, payment_intent: 'pi_ref_1' });
    const lost = putDispute('dp_both', { status: 'lost', charge: 'ch_both', payment_intent: 'pi_ref_1', amount: 1000, customer: 'cus_1' });
    const [rf, dp] = await Promise.all([
      worker.fetch(await signed(refundEvt('evt_both_rf', charge)), env),
      worker.fetch(await signed(disputeEvt('evt_both_dp', 'charge.dispute.closed', lost)), env),
    ]);
    expect(rf.status).toBe(200);
    expect(dp.status).toBe(200);
    const clawed = (await rf.json()).clawed + (await dp.json()).clawed;
    expect(clawed).toBe(10000);
    expect(grants()[0].clawed_credits).toBe(10000);
    expect(bal()).toBe(1000);
    expect(bal()).toBeGreaterThanOrEqual(0);
    assertLedgerMatchesBalance();
  });

  it('a claw while a hold is open is capped at available; settleHold afterwards leaves balance >= 0', async () => {
    await grantPack();
    expect(await placeHold(env.DB, 'hold_open', 'u1', 2000)).toBe(true);
    const charge = putCharge('ch_hold', { amount: 1000, amount_refunded: 1000, payment_intent: 'pi_ref_1' });
    const j = await (await worker.fetch(await signed(refundEvt('evt_rf_hold', charge)), env)).json();
    expect(j.clawed).toBe(9000);
    expect(j.shortfall).toBe(1000);
    expect(j.frozen).toBe(true);
    expect(bal()).toBe(2000);
    expect(unmatched()).toMatchObject([{ reason: 'refund_claw_shortfall', detail: '1000' }]);
    const charged = await settleHold(env.DB, 'hold_open', 15);
    expect(charged).toBe(15);
    expect(bal()).toBe(1985);
    expect(bal()).toBeGreaterThanOrEqual(0);
    assertLedgerMatchesBalance();
  });

  it('frozen account cannot start checkout (403 before Stripe)', async () => {
    await grantPack();
    env.DB.q("UPDATE accounts SET frozen = 1, freeze_reason = 'dispute' WHERE id='u1'");
    const raw = ['sess', crypto.randomUUID().replace(/-/g, ''), 'token'].join('');
    env.DB.q(
      'INSERT INTO sessions (session_hash, account_id, expires_at) VALUES (?1, ?2, ?3)',
      await sha256Hex(raw),
      'u1',
      Math.floor(Date.now() / 1000) + 86400,
    );
    const f = stripeFetch();
    const r = await worker.fetch(new Request('https://api.rathor.ai/v1/checkout', {
      method: 'POST',
      headers: {
        Origin: O,
        'Content-Type': 'application/json',
        Cookie: `${SESSION_COOKIE_NAME}=${raw}`,
        [CSRF_HEADER]: '1',
      },
      body: JSON.stringify({ price: 'pack_10' }),
    }), env);
    expect(r.status).toBe(403);
    expect(await r.json()).toMatchObject({ error: 'account_frozen' });
    expect(f.mock.calls.filter((c) => String(c[0]).includes('/v1/checkout/sessions'))).toHaveLength(0);
    assertLedgerMatchesBalance();
  });

  it('GET /v1/credits sends Cache-Control no-store', async () => {
    const r = await worker.fetch(creditsReq(), env);
    expect(r.status).toBe(200);
    expect(r.headers.get('Cache-Control')).toBe('no-store');
    expect(await r.json()).toMatchObject({ credits: 1000, balance: 1000, frozen: false });
    assertLedgerMatchesBalance();
  });
});

describe('SPROOF-FIX-DISPUTE-ORDER early dispute (dispute before grant)', () => {
  const early = () => env.DB.q('SELECT * FROM early_disputes');
  const events = (id) => env.DB.q('SELECT * FROM stripe_events WHERE event_id = ?1', id);
  const grantAfter = async (id = 'evt_g_late', pi = 'pi_early') => {
    const e = checkoutEvt({ id });
    e.data.object.payment_intent = pi;
    e.data.object.customer = 'cus_new';
    return (await worker.fetch(await signed(e), env)).json();
  };

  it('dispute.created before the grant is stored durably, answers 503 while fresh, and the grant freezes in the same batch', async () => {
    stripeFetch();
    const d = putDispute('dp_e1', { status: 'needs_response', charge: 'ch_e1', payment_intent: 'pi_early', customer: 'cus_new', amount: 1000 });
    const r = await worker.fetch(await signed({ ...disputeEvt('evt_dp_e1', 'charge.dispute.created', d), created: Math.floor(Date.now() / 1000) }), env);
    expect(r.status).toBe(503);
    expect(await r.json()).toMatchObject({ error: 'dispute_grant_pending', retry: true, stored: true });
    expect(early()).toMatchObject([{ dispute_id: 'dp_e1', status: 'needs_response', payment_intent_id: 'pi_early', charge_id: 'ch_e1', applied_event_id: null }]);
    expect(events('evt_dp_e1')).toHaveLength(0);
    expect(frozen().frozen).toBe(0);

    // The early-dispute statements ride in the same batch as the credit_grants INSERT.
    const db = env.DB;
    const realBatch = db.batch.bind(db);
    db.batch = async (stmts) => {
      if (stmts.some((s) => /INSERT INTO credit_grants/.test(s.sql))) {
        expect(stmts.some((s) => /early_disputes/.test(s.sql))).toBe(true);
      }
      return realBatch(stmts);
    };
    const j = await grantAfter();
    expect(j.granted).toBe(10000);
    expect(j.frozen).toBe(true);
    expect(j.early_disputes).toEqual([{ dispute_id: 'dp_e1', status: 'needs_response' }]);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute' });
    expect(bal()).toBe(11000);
    expect(early()[0]).toMatchObject({ applied_event_id: 'evt_g_late', applied_account_id: 'u1' });
    expect(env.DB.q('SELECT * FROM stripe_disputes')).toMatchObject([{ dispute_id: 'dp_e1', account_id: 'u1', status: 'needs_response' }]);
    expect((await worker.fetch(chatReq(), env)).status).toBe(403);

    // Stripe's retry now matches the grant: still frozen, no double effect.
    const r2 = await worker.fetch(await signed(disputeEvt('evt_dp_e1', 'charge.dispute.created', d)), env);
    expect(r2.status).toBe(200);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute' });
    // A later won close unfreezes through the normal path.
    const won = putDispute('dp_e1', { status: 'won', charge: 'ch_e1', payment_intent: 'pi_early', customer: 'cus_new' });
    const w = await (await worker.fetch(await signed(disputeEvt('evt_dp_e1w', 'charge.dispute.closed', won)), env)).json();
    expect(w.frozen).toBe(false);
    assertLedgerMatchesBalance();
  });

  it('grant batch is atomic: if it fails, neither the grant nor the early-dispute freeze is applied', async () => {
    stripeFetch();
    const d = putDispute('dp_atom', { status: 'needs_response', charge: 'ch_atom', payment_intent: 'pi_early' });
    await worker.fetch(await signed(disputeEvt('evt_dp_atom', 'charge.dispute.created', d)), env);
    const db = env.DB;
    const realBatch = db.batch.bind(db);
    db.batch = async (stmts) => {
      if (stmts.some((s) => /INSERT INTO credit_grants/.test(s.sql))) {
        return realBatch([...stmts, db.prepare('INSERT INTO no_such_table VALUES (1)')]);
      }
      return realBatch(stmts);
    };
    const e = checkoutEvt({ id: 'evt_g_atom' });
    e.data.object.payment_intent = 'pi_early';
    let status = 0;
    try { status = (await worker.fetch(await signed(e), env)).status; } catch { status = 500; }
    expect(status).toBeGreaterThanOrEqual(500);
    db.batch = realBatch;
    expect(grants().filter((g) => g.payment_intent_id === 'pi_early')).toHaveLength(0);
    expect(frozen().frozen).toBe(0);
    expect(early()[0].applied_event_id).toBe(null);
  });

  it('dispute closed as won before the grant: no freeze at grant', async () => {
    stripeFetch();
    const c = putDispute('dp_ew', { status: 'needs_response', charge: 'ch_ew', payment_intent: 'pi_early' });
    expect((await worker.fetch(await signed(disputeEvt('evt_dp_ew1', 'charge.dispute.created', c)), env)).status).toBe(503);
    const w = putDispute('dp_ew', { status: 'won', charge: 'ch_ew', payment_intent: 'pi_early' });
    expect((await worker.fetch(await signed(disputeEvt('evt_dp_ew2', 'charge.dispute.closed', w)), env)).status).toBe(503);
    // A late replay of created must not reopen it (terminal status is sticky).
    await worker.fetch(await signed(disputeEvt('evt_dp_ew1', 'charge.dispute.created', putDispute('dp_ew', { status: 'needs_response', charge: 'ch_ew', payment_intent: 'pi_early' }))), env);
    expect(early()).toMatchObject([{ dispute_id: 'dp_ew', status: 'won' }]);
    const j = await grantAfter();
    expect(j.granted).toBe(10000);
    expect(j.frozen).toBe(false);
    expect(frozen()).toMatchObject({ frozen: 0, freeze_reason: null });
    expect(bal()).toBe(11000);
    assertLedgerMatchesBalance();
  });

  it('dispute closed as lost before the grant: same claw + freeze as the current lost path', async () => {
    // Reference: current lost path (grant first, then lost).
    await grantPack();
    const open = putDispute('dp_ref', { status: 'needs_response', charge: 'ch_ref', payment_intent: 'pi_ref_1', customer: 'cus_1' });
    await worker.fetch(await signed(disputeEvt('evt_ref1', 'charge.dispute.created', open)), env);
    const lost = putDispute('dp_ref', { status: 'lost', charge: 'ch_ref', payment_intent: 'pi_ref_1', amount: 1000, customer: 'cus_1' });
    await worker.fetch(await signed(disputeEvt('evt_ref2', 'charge.dispute.closed', lost)), env);
    const ref = { bal: bal(), frozen: frozen(), clawed: grants()[0].clawed_credits };

    // Fresh DB: dispute lost before the grant.
    env.DB = fakeD1();
    env.DB.q("INSERT INTO accounts (id, balance) VALUES ('u1', 1000)");
    env.DB.q("INSERT INTO ledger (account_id, delta, reason) VALUES ('u1', 1000, 'seed')");
    env.DB.q('INSERT INTO api_keys (key_hash, account_id) VALUES (?1, ?2)', await sha256Hex(KEY), 'u1');
    const l = putDispute('dp_el', { status: 'lost', charge: 'ch_el', payment_intent: 'pi_early', amount: 1000 });
    expect((await worker.fetch(await signed(disputeEvt('evt_dp_el', 'charge.dispute.closed', l)), env)).status).toBe(503);
    const j = await grantAfter();
    expect(j.granted).toBe(10000);
    expect(j.frozen).toBe(true);
    expect({ bal: bal(), frozen: frozen(), clawed: grants()[0].clawed_credits }).toEqual(ref);
    expect(ref).toEqual({ bal: 1000, frozen: { frozen: 1, freeze_reason: 'dispute_lost' }, clawed: 10000 });
    expect(env.DB.q("SELECT delta FROM ledger WHERE reason = 'dispute_lost_claw'")).toEqual([{ delta: -10000 }]);
    expect(env.DB.q('SELECT status FROM stripe_disputes')).toEqual([{ status: 'lost' }]);
    expect((await worker.fetch(chatReq(), env)).status).toBe(403);
    assertLedgerMatchesBalance();
  });

  it('replayed dispute event is idempotent: one early row, one freeze, no double claw', async () => {
    stripeFetch();
    const l = putDispute('dp_rep', { status: 'lost', charge: 'ch_rep', payment_intent: 'pi_early', amount: 1000 });
    for (let i = 0; i < 3; i++) {
      expect((await worker.fetch(await signed(disputeEvt('evt_dp_rep', 'charge.dispute.closed', l)), env)).status).toBe(503);
    }
    expect(early()).toHaveLength(1);
    expect(early()[0]).toMatchObject({ dispute_id: 'dp_rep', stripe_event_id: 'evt_dp_rep', status: 'lost' });
    await grantAfter();
    expect(bal()).toBe(1000);
    // Replays after the grant go through the normal lost path: already clawed → nothing more.
    for (let i = 0; i < 2; i++) {
      expect((await worker.fetch(await signed(disputeEvt('evt_dp_rep', 'charge.dispute.closed', l)), env)).status).toBe(200);
    }
    expect(bal()).toBe(1000);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE reason = 'dispute_lost_claw'")).toEqual([{ n: 1 }]);
    expect(grants()[0].clawed_credits).toBe(10000);
    expect(early()).toHaveLength(1);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute_lost' });
    // Replaying the grant event is still a duplicate and changes nothing.
    const dup = await grantAfter();
    expect(dup).toMatchObject({ duplicate: true });
    expect(bal()).toBe(1000);
    assertLedgerMatchesBalance();
  });

  it('an exhausted (old) unmatched dispute is still stored and then recorded once with HTTP 200', async () => {
    stripeFetch();
    const d = putDispute('dp_old', { status: 'needs_response', charge: 'ch_old', payment_intent: 'pi_early' });
    const r = await worker.fetch(await signed({ ...disputeEvt('evt_dp_old', 'charge.dispute.created', d), created: Math.floor(Date.now() / 1000) - 262800 }), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'dispute_no_matching_grant' });
    expect(early()).toHaveLength(1);
    const j = await grantAfter();
    expect(j.frozen).toBe(true);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute' });
  });
});

describe('SPROOF-FIX-DISPUTE-ORDER production exhaust', () => {
  it('production STRIPE_RETRY_EXHAUST_SECONDS >= 259200 + margin (and code default matches)', async () => {
    const { readFileSync } = await import('node:fs');
    const toml = readFileSync(new URL('../wrangler.toml', import.meta.url), 'utf8');
    const prod = toml.split('[env.production.vars]')[1];
    const v = Number(prod.match(/STRIPE_RETRY_EXHAUST_SECONDS = "(\d+)"/)[1]);
    expect(v).toBeGreaterThanOrEqual(259200 + 3600);
    expect(v).toBe(262800);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: String(v) })).toBe(v);
    expect(STRIPE_RETRY_EXHAUST_SECONDS_DEFAULT).toBeGreaterThanOrEqual(259200 + 3600);
    // Sandbox stays 21600.
    expect(toml.split('\n[env.production]')[0]).toMatch(/STRIPE_RETRY_EXHAUST_SECONDS = "21600"/);
  });
});

describe('SPROOF-FIX-DISPUTE-RACE account found by customer, no grant yet', () => {
  const early = () => env.DB.q('SELECT * FROM early_disputes');
  const events = (id) => env.DB.q('SELECT * FROM stripe_events WHERE event_id = ?1', id);
  const now = () => Math.floor(Date.now() / 1000);
  const evt = (id, type, d, created = now()) => ({ ...disputeEvt(id, type, d), created });
  const grantRace = async (id = 'evt_g_race') => {
    const e = checkoutEvt({ id });
    e.data.object.payment_intent = 'pi_race';
    return (await worker.fetch(await signed(e), env)).json();
  };
  beforeEach(() => { env.DB.q("UPDATE accounts SET stripe_customer_id = 'cus_1' WHERE id = 'u1'"); });

  it('fresh: 503 dispute_grant_pending, stored row, no freeze; then the grant freezes in the same batch', async () => {
    stripeFetch();
    const d = putDispute('dp_r1', { status: 'needs_response', charge: 'ch_r1', payment_intent: 'pi_race', customer: 'cus_1', amount: 1000 });
    const r = await worker.fetch(await signed(evt('evt_dp_r1', 'charge.dispute.created', d)), env);
    expect(r.status).toBe(503);
    expect(await r.json()).toMatchObject({ error: 'dispute_grant_pending', retry: true, stored: true });
    expect(early()).toMatchObject([{ dispute_id: 'dp_r1', status: 'needs_response', applied_event_id: null }]);
    expect(events('evt_dp_r1')).toHaveLength(0);
    expect(frozen()).toMatchObject({ frozen: 0 });
    expect(env.DB.q('SELECT * FROM stripe_disputes')).toHaveLength(0);

    const db = env.DB;
    const realBatch = db.batch.bind(db);
    let sameBatch = false;
    db.batch = async (stmts) => {
      if (stmts.some((s) => /INSERT INTO credit_grants/.test(s.sql))) sameBatch = stmts.some((s) => /early_disputes/.test(s.sql));
      return realBatch(stmts);
    };
    const j = await grantRace();
    expect(sameBatch).toBe(true);
    expect(j).toMatchObject({ granted: 10000, frozen: true });
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute' });
    expect(early()[0]).toMatchObject({ applied_event_id: 'evt_g_race', applied_account_id: 'u1' });
    // Stripe's retry now finds the grant: 200, still frozen.
    const r2 = await worker.fetch(await signed(evt('evt_dp_r1', 'charge.dispute.created', d)), env);
    expect(r2.status).toBe(200);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute' });
    assertLedgerMatchesBalance();
  });

  it('lost dispute before the grant: after the grant the claw and freeze match the current lost path', async () => {
    stripeFetch();
    const l = putDispute('dp_rl', { status: 'lost', charge: 'ch_rl', payment_intent: 'pi_race', customer: 'cus_1', amount: 1000 });
    expect((await worker.fetch(await signed(evt('evt_dp_rl', 'charge.dispute.closed', l)), env)).status).toBe(503);
    expect(frozen()).toMatchObject({ frozen: 0 });
    expect(bal()).toBe(1000);
    const j = await grantRace();
    expect(j).toMatchObject({ granted: 10000, frozen: true });
    // Current lost path (grant then lost) leaves balance 1000, clawed 10000, frozen dispute_lost.
    expect(bal()).toBe(1000);
    expect(grants()[0].clawed_credits).toBe(10000);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute_lost' });
    expect(env.DB.q("SELECT delta FROM ledger WHERE reason = 'dispute_lost_claw'")).toEqual([{ delta: -10000 }]);
    // Retry after the grant: normal lost path, nothing more to claw.
    const r = await worker.fetch(await signed(evt('evt_dp_rl', 'charge.dispute.closed', l)), env);
    expect(r.status).toBe(200);
    expect(bal()).toBe(1000);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE reason = 'dispute_lost_claw'")).toEqual([{ n: 1 }]);
    assertLedgerMatchesBalance();
  });

  it('exhausted old event: stored once, existing 200 path, no infinite retry', async () => {
    stripeFetch();
    const d = putDispute('dp_rold', { status: 'needs_response', charge: 'ch_rold', payment_intent: 'pi_race', customer: 'cus_1' });
    const old = now() - 262800;
    const r = await worker.fetch(await signed(evt('evt_dp_rold', 'charge.dispute.created', d, old)), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ frozen: true, dispute_id: 'dp_rold' });
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute' });
    expect(events('evt_dp_rold')).toHaveLength(1);
    expect(early()).toHaveLength(1);
    const again = await worker.fetch(await signed(evt('evt_dp_rold', 'charge.dispute.created', d, old)), env);
    expect(again.status).toBe(200);
    expect(await again.json()).toMatchObject({ duplicate: true });
    expect(early()).toHaveLength(1);
  });

  it('replay is idempotent: one early row, no stripe_events row, one freeze and claw at grant', async () => {
    stripeFetch();
    const l = putDispute('dp_rrep', { status: 'lost', charge: 'ch_rrep', payment_intent: 'pi_race', customer: 'cus_1', amount: 1000 });
    for (let i = 0; i < 3; i++) {
      expect((await worker.fetch(await signed(evt('evt_dp_rrep', 'charge.dispute.closed', l)), env)).status).toBe(503);
    }
    expect(early()).toHaveLength(1);
    expect(events('evt_dp_rrep')).toHaveLength(0);
    expect(frozen()).toMatchObject({ frozen: 0 });
    await grantRace();
    expect((await grantRace()).duplicate).toBe(true);
    expect(bal()).toBe(1000);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE reason = 'dispute_lost_claw'")).toEqual([{ n: 1 }]);
    assertLedgerMatchesBalance();
  });
});

// FREEZE-PRECEDENCE-1: settle_over_hold lives in accounts.settle_locked, never in freeze_reason.
describe('FREEZE-PRECEDENCE-1 dispute freezes never masked by settle_over_hold', () => {
  const lockRow = (id = 'u1') => env.DB.q('SELECT frozen, freeze_reason, settle_locked FROM accounts WHERE id = ?1', id)[0];
  // A small shortfall that keeps the balance: hold A (10) settles at 20 while hold B reserves
  // every other credit, so 10 is charged and 10 is recorded short. B is then released.
  async function shortSettle(tag = 'x') {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const avail = bal() - (env.DB.q("SELECT COALESCE(SUM(amount),0) AS s FROM holds WHERE account_id='u1' AND status='open'")[0].s);
    expect(await placeHold(env.DB, `A_${tag}`, 'u1', 10)).toBe(true);
    expect(await placeHold(env.DB, `B_${tag}`, 'u1', avail - 10)).toBe(true);
    expect(await settleHold(env.DB, `A_${tag}`, 20)).toBe(10);
    await releaseHold(env.DB, `B_${tag}`);
    expect(env.DB.q('SELECT shortfall_credits FROM settle_shortfalls WHERE hold_id = ?1', `A_${tag}`)[0].shortfall_credits).toBe(10);
  }
  const dispute = async (id, status, evtId, type = 'charge.dispute.created') => {
    const d = putDispute(id, { status, charge: `ch_${id}`, payment_intent: 'pi_ref_1', customer: 'cus_1', amount: 1000 });
    return (await worker.fetch(await signed(disputeEvt(evtId, type, d)), env)).json();
  };
  const chatOk = () => { vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }], usage: { prompt_tokens: 1, completion_tokens: 1 } }))); };

  it('settle_over_hold then dispute: dispute lands in freeze_reason, settle lock kept', async () => {
    await grantPack();
    await shortSettle();
    expect(lockRow()).toEqual({ frozen: 0, freeze_reason: null, settle_locked: 1 });
    await dispute('dp_p1', 'needs_response', 'evt_p1');
    expect(lockRow()).toEqual({ frozen: 1, freeze_reason: 'dispute', settle_locked: 1 });
    expect((await worker.fetch(chatReq(), env)).status).toBe(403);
  });
  it('dispute then settle_over_hold (call in flight): dispute reason is not overwritten', async () => {
    await grantPack();
    vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(await placeHold(env.DB, 'inflight', 'u1', 10)).toBe(true);
    expect(await placeHold(env.DB, 'rest', 'u1', bal() - 10)).toBe(true);
    await dispute('dp_p2', 'needs_response', 'evt_p2');
    expect(await settleHold(env.DB, 'inflight', 20)).toBe(10);
    await releaseHold(env.DB, 'rest');
    expect(lockRow()).toEqual({ frozen: 1, freeze_reason: 'dispute', settle_locked: 1 });
  });
  it('dispute_lost both orders keeps dispute_lost and the settle lock', async () => {
    await grantPack();
    await shortSettle('a');
    await dispute('dp_l', 'needs_response', 'evt_l1');
    await dispute('dp_l', 'lost', 'evt_l2', 'charge.dispute.closed');
    expect(lockRow()).toEqual({ frozen: 1, freeze_reason: 'dispute_lost', settle_locked: 1 });
    // later over-hold settle (in flight before the freeze) does not touch dispute_lost
    vi.spyOn(console, 'error').mockImplementation(() => {});
    env.DB.q("INSERT INTO holds (id, account_id, amount, status) VALUES ('late', 'u1', 1, 'open')");
    await settleHold(env.DB, 'late', 1_000_000);
    expect(lockRow()).toMatchObject({ frozen: 1, freeze_reason: 'dispute_lost', settle_locked: 1 });
  });
  it('dispute_lost first, then a short settle in flight: dispute_lost stays', async () => {
    await grantPack();
    vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(await placeHold(env.DB, 'f', 'u1', 10)).toBe(true);
    await dispute('dp_l3', 'needs_response', 'evt_l3');
    await dispute('dp_l3', 'lost', 'evt_l4', 'charge.dispute.closed');
    const before = lockRow().freeze_reason;
    expect(['dispute_lost', 'refund_claw_shortfall']).toContain(before);
    await settleHold(env.DB, 'f', 1_000_000);
    expect(lockRow()).toMatchObject({ frozen: 1, freeze_reason: before, settle_locked: 1 });
  });
  it('unlock settle_over_hold with an active dispute: lock cleared, dispute freeze stays (403)', async () => {
    await grantPack();
    await shortSettle();
    await dispute('dp_u1', 'needs_response', 'evt_u1');
    const r = await unlockSettleOverHold(env.DB, 'u1');
    expect(r).toEqual({ unlocked: true, still_frozen: true, freeze_reason: 'dispute' });
    expect(lockRow()).toEqual({ frozen: 1, freeze_reason: 'dispute', settle_locked: 0 });
    expect(env.DB.q('SELECT resolved FROM settle_shortfalls')[0].resolved).toBe(1);
    const c = await worker.fetch(chatReq(), env);
    expect(c.status).toBe(403); expect(await c.json()).toMatchObject({ error: 'account_frozen' });
  });
  it('unlock settle_over_hold without a dispute: account usable again', async () => {
    await grantPack();
    await shortSettle();
    expect((await worker.fetch(chatReq(), env)).status).toBe(403);
    expect(await unlockSettleOverHold(env.DB, 'u1')).toEqual({ unlocked: true, still_frozen: false, freeze_reason: null });
    vi.restoreAllMocks(); chatOk();
    expect((await worker.fetch(chatReq(), env)).status).toBe(200);
  });
  it('dispute won / warning_closed with an active settle lock: dispute unfreezes, settle lock keeps it locked', async () => {
    for (const st of ['won', 'warning_closed']) {
      vi.restoreAllMocks();
      env.DB.q("UPDATE accounts SET frozen = 0, freeze_reason = NULL, settle_locked = 0 WHERE id='u1'");
      if (st === 'won') await grantPack(); else stripeFetch();
      await shortSettle(st);
      await dispute(`dp_${st}`, 'needs_response', `evt_${st}_1`);
      const j = await dispute(`dp_${st}`, st, `evt_${st}_2`, 'charge.dispute.closed');
      expect(j).toMatchObject({ frozen: true, dispute_status: st });
      expect(lockRow()).toEqual({ frozen: 0, freeze_reason: null, settle_locked: 1 });
      vi.restoreAllMocks(); chatOk();
      expect((await worker.fetch(chatReq(), env)).status).toBe(403);
      const read = await (await worker.fetch(creditsReq(), env)).json();
      expect(read.frozen).toBe(true);
    }
  });
  it('unapplied early dispute matching the account by payment intent or charge blocks the unlock', async () => {
    await grantPack();
    await shortSettle();
    env.DB.q("INSERT INTO early_disputes (dispute_id, status, payment_intent_id, charge_id, stripe_event_id) VALUES ('dp_e', 'needs_response', 'pi_ref_1', NULL, 'evt_e')");
    env.DB.q("INSERT INTO early_disputes (dispute_id, status, payment_intent_id, charge_id, stripe_event_id) VALUES ('dp_other', 'needs_response', 'pi_someone_else', NULL, 'evt_o')");
    env.DB.q("INSERT INTO early_disputes (dispute_id, status, payment_intent_id, charge_id, stripe_event_id) VALUES ('dp_won_e', 'won', 'pi_ref_1', NULL, 'evt_w')");
    expect((await pendingEarlyDisputes(env.DB, 'u1')).map((r) => r.dispute_id)).toEqual(['dp_e']);
    expect(await unlockSettleOverHold(env.DB, 'u1')).toEqual({ unlocked: false, reason: 'unapplied_early_dispute', disputes: ['dp_e'] });
    expect(lockRow().settle_locked).toBe(1);
    env.DB.q("UPDATE early_disputes SET status = 'won' WHERE dispute_id = 'dp_e'");
    expect((await unlockSettleOverHold(env.DB, 'u1')).unlocked).toBe(true);
  });
  it('refund_claw_shortfall is kept by a later dispute and untouched by a settle lock', async () => {
    await grantPack();
    env.DB.q("UPDATE accounts SET frozen = 1, freeze_reason = 'refund_claw_shortfall' WHERE id='u1'");
    env.DB.q("INSERT INTO holds (id, account_id, amount, status) VALUES ('rc', 'u1', 1, 'open')");
    vi.spyOn(console, 'error').mockImplementation(() => {});
    await settleHold(env.DB, 'rc', 1_000_000);
    await dispute('dp_rc', 'needs_response', 'evt_rc');
    expect(lockRow()).toEqual({ frozen: 1, freeze_reason: 'refund_claw_shortfall', settle_locked: 1 });
  });
});

describe('FREEZE-PRECEDENCE-1 migration 0009 backfill', () => {
  it('restores a masked dispute / dispute_lost, sets settle_locked, and never unlocks', async () => {
    const { readFileSync } = await import('node:fs');
    const sql = readFileSync(new URL('../migrations/0009_settle_lock.sql', import.meta.url), 'utf8')
      .split('\n').filter((l) => !/ALTER TABLE/.test(l) && !/^\s*--/.test(l)).join('\n');
    env.DB.q("INSERT INTO accounts (id, balance, frozen, freeze_reason) VALUES ('m_open', 0, 1, 'settle_over_hold'), ('m_lost', 0, 1, 'settle_over_hold'), ('m_none', 0, 1, 'settle_over_hold'), ('m_won', 0, 1, 'settle_over_hold'), ('m_disp', 0, 1, 'dispute'), ('m_short', 0, 0, NULL), ('m_claw', 0, 1, 'settle_over_hold'), ('m_claw_d', 0, 1, 'settle_over_hold')");
    env.DB.q("INSERT INTO unmatched_payments (stripe_event_id, event_type, reason, account_id, mode, detail) VALUES ('evt_c1', 'charge.refunded', 'refund_claw_shortfall', 'm_claw', 'charge', '500'), ('evt_c2', 'charge.dispute.closed', 'refund_claw_shortfall', 'm_claw_d', 'charge', '500'), ('evt_c3', 'charge.refunded', 'unknown_price', 'm_none', 'charge', NULL)");
    env.DB.q("INSERT INTO stripe_disputes (dispute_id, account_id, status) VALUES ('d5','m_claw_d','lost')");
    env.DB.q("INSERT INTO stripe_disputes (dispute_id, account_id, status) VALUES ('d1','m_open','needs_response'), ('d2','m_lost','lost'), ('d3','m_won','won'), ('d4','m_disp','needs_response')");
    env.DB.q("INSERT INTO settle_shortfalls (account_id, hold_id, actual_credits, charged_credits, shortfall_credits) VALUES ('m_short','h1',5,1,4)");
    for (const stmt of sql.split(';').map((x) => x.trim()).filter(Boolean)) env.DB.q(stmt);
    const r = (id) => env.DB.q('SELECT frozen, freeze_reason, settle_locked FROM accounts WHERE id = ?1', id)[0];
    expect(r('m_open')).toEqual({ frozen: 1, freeze_reason: 'dispute', settle_locked: 1 });
    expect(r('m_lost')).toEqual({ frozen: 1, freeze_reason: 'dispute_lost', settle_locked: 1 });
    expect(r('m_none')).toEqual({ frozen: 0, freeze_reason: null, settle_locked: 1 });
    expect(r('m_won')).toEqual({ frozen: 0, freeze_reason: null, settle_locked: 1 });
    expect(r('m_disp')).toEqual({ frozen: 1, freeze_reason: 'dispute', settle_locked: 0 });
    expect(r('m_short')).toEqual({ frozen: 0, freeze_reason: null, settle_locked: 1 });
    // Clerk #760: a refund claw shortfall hidden under settle_over_hold is restored first.
    expect(r('m_claw')).toEqual({ frozen: 1, freeze_reason: 'refund_claw_shortfall', settle_locked: 1 });
    expect(r('m_claw_d')).toEqual({ frozen: 1, freeze_reason: 'refund_claw_shortfall', settle_locked: 1 });
    // After a settle unlock the restored claw freeze still blocks.
    expect((await unlockSettleOverHold(env.DB, 'm_claw')).still_frozen).toBe(true);
    expect(await placeHold(env.DB, 'mc', 'm_claw', 1)).toBe(false);
    expect(await placeHold(env.DB, 'mh', 'm_none', 0 + 1)).toBe(false);
  });
});
