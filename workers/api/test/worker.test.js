import { describe, it, expect, vi, beforeEach } from 'vitest';
import worker from '../src/index.js';
import { hmacHex, verifyStripeSignature } from '../src/stripe.js';
import { creditsForUsage, creditsForTokens, creditsToUsd, maxCostEstimate, holdEstimate, RATES_VERIFIED, MARKUP, USD_PER_CREDIT, CREDIT_PACKS, MIN_PACK_USD, PRICING, XAI_RATES_USD_PER_MTOK } from '../src/credits.js';
import { sha256Hex, constantTimeEqual } from '../src/auth.js';
import { mercyChecks, MERCY_CHECKS_BASELINE } from '../src/mercyChecks.js';
import { MAX_TOKENS_CAP, DEFAULT_MAX_TOKENS, ALLOWED_MODELS, DEFAULT_MODEL, isLiveMode, stripeKeyMatchesLiveMode, stripeRetryExhaustSeconds, STRIPE_RETRY_EXHAUST_SECONDS_DEFAULT } from '../src/config.js';
import { fakeD1 } from './fakeD1.js';
import { isSchemaMissing } from '../src/db.js';

// Placeholders built at runtime so no key literal appears in the test file.
const sk = (mode, tail = 'PLACEHOLDER') => ['sk', mode, tail].join('_');
const rk = (mode, tail = 'PLACEHOLDER') => ['rk', mode, tail].join('_');
const SK_TEST = sk('test');
const SK_LIVE = sk('live');
const RK_TEST = rk('test');
const RK_LIVE = rk('live');

const O = 'https://rathor.ai';
const PRICES = { STRIPE_PRICE_PACK_10: 'price_test_p10', STRIPE_PRICE_PACK_25: 'price_test_p25', STRIPE_PRICE_PACK_50: 'price_test_p50', STRIPE_PRICE_PLAN_MONTHLY: 'price_test_plan', STRIPE_PRICE_TRIAL: 'price_test_trial' };
const UNIT = { price_test_p10: 1000, price_test_p25: 2500, price_test_p50: 5000, price_test_plan: 1000, price_test_trial: 100 };
// Mocked Stripe line item list (shape per Stripe API docs for GET /v1/checkout/sessions/{id}/line_items;
// not verified against a live test-mode response).
export const li = (id, { qty = 1, unit = UNIT[id] ?? 1000, currency = 'usd', subtotal, discount = 0, tax = 0 } = {}) => {
  const sub = subtotal ?? unit * qty;
  return { object: 'list', has_more: false, data: [{ id: 'li_test', object: 'item', quantity: qty, currency, amount_subtotal: sub, amount_discount: discount, amount_tax: tax, amount_total: sub - discount + tax, price: { id, object: 'price', unit_amount: unit, currency } }] };
};
const LINE_ITEMS = new Map();
const KEY = ['rth', 'test', '0123456789abcdefghijklmnop'].join('_');
let env;
beforeEach(async () => {
  vi.restoreAllMocks();
  env = { ALLOWED_ORIGIN: O, DB: fakeD1(), XAI_API_KEY: 'test', STRIPE_WEBHOOK_SECRET: 'whsec_test', RESEND_API_KEY: 'test', RATES_VERIFIED: 'true', STRIPE_SECRET_KEY: 'sk_test_FAKE_not_a_real_key', ...PRICES };
  LINE_ITEMS.clear();
  env.DB.q("INSERT INTO accounts (id, balance) VALUES ('u1', 1000)");
  env.DB.q('INSERT INTO api_keys (key_hash, account_id) VALUES (?1, ?2)', await sha256Hex(KEY), 'u1');
});
const bal = (id = 'u1') => env.DB.q('SELECT balance FROM accounts WHERE id = ?1', id)[0]?.balance;

async function signed(evt, secret = 'whsec_test') {
  // Line items are not in the webhook payload; register them for the mocked Stripe API instead.
  const ob = evt.data?.object;
  if (evt.type === 'checkout.session.completed' && ob && 'line_items' in ob) { evt = structuredClone(evt); LINE_ITEMS.set(evt.data.object.id, evt.data.object.line_items); delete evt.data.object.line_items; }
  const body = JSON.stringify(evt); const t = Math.floor(Date.now() / 1000);
  return new Request('https://api.rathor.ai/v1/stripe/webhook', { method: 'POST', body, headers: { 'Stripe-Signature': `t=${t},v1=${await hmacHex(secret, `${t}.${body}`)}` } });
}
const evt = (o = {}) => { const e = { id: 'evt_1', type: 'checkout.session.completed', livemode: false, data: { object: { line_items: li('price_test_p10'), client_reference_id: 'u1', mode: 'payment', payment_status: 'paid', amount_total: 1000, currency: 'usd', metadata: {}, customer_details: { email: 'test@example.com' } } }, ...o }; e.data.object.id ??= `cs_test_${e.id}`; return e; };
// Global fetch mock: Stripe line items from LINE_ITEMS (404 if unknown), everything else '{}'.
const stripeFetch = (onLineItems) => vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
  const m = String(url).match(/^https:\/\/api\.stripe\.com\/v1\/checkout\/sessions\/([^/]+)\/line_items/);
  if (m) {
    if (onLineItems) { const r = await onLineItems(decodeURIComponent(m[1]), init); if (r) return r; }
    const l = LINE_ITEMS.get(decodeURIComponent(m[1]));
    return l ? new Response(JSON.stringify(l)) : new Response('{"error":{"type":"invalid_request_error"}}', { status: 404 });
  }
  return new Response('{}');
});
const okFetch = (usage = { prompt_tokens: 1000, completion_tokens: 1000 }, content = 'hello') =>
  vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(JSON.stringify({ choices: [{ message: { content } }], usage })));

describe('health', () => {
  it('ok', async () => { const r = await worker.fetch(new Request('https://api.rathor.ai/health'), env); expect(r.status).toBe(200); });
});

describe('stripe signature', () => {
  it('rejects bad and stale', async () => {
    expect(await verifyStripeSignature('x', 't=1,v1=abc', 's')).toBe(false);
    const t = 1000; const sig = await hmacHex('s', `${t}.x`);
    expect(await verifyStripeSignature('x', `t=${t},v1=${sig}`, 's', 1000)).toBe(true);
    expect(await verifyStripeSignature('x', `t=${t},v1=${sig}`, 's', 2000)).toBe(false);
  });
});

describe('webhook (D1, idempotent)', () => {
  it('grants credits once per event id and emails from info@rathor.ai', async () => {
    const f = stripeFetch();
    const j = await (await worker.fetch(await signed(evt()), env)).json();
    expect(j.balance).toBe(11000);
    const resend = f.mock.calls.find((c) => String(c[0]).includes('resend'));
    expect(JSON.parse(resend[1].body).from).toBe('Ra-Thor <info@rathor.ai>');
    const again = await (await worker.fetch(await signed(evt()), env)).json();
    expect(again.duplicate).toBe(true);
    expect(bal()).toBe(11000);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_1'")[0].n).toBe(1);
  });
  it('concurrent duplicate deliveries grant once', async () => {
    stripeFetch();
    await Promise.all([1, 2, 3].map(async () => worker.fetch(await signed(evt()), env)));
    expect(bal()).toBe(11000);
  });
  it('unique constraint on event id', () => {
    env.DB.q("INSERT INTO stripe_events (event_id, type) VALUES ('e', 't')");
    expect(() => env.DB.q("INSERT INTO stripe_events (event_id, type) VALUES ('e', 't')")).toThrow(/UNIQUE/);
  });
  it('LIVE-GATE-1 (b): live event in sandbox is 200 ignored livemode_mismatch, recorded, 0 credits', async () => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const r = await worker.fetch(await signed(evt({ id: 'evt_live_b', livemode: true })), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'livemode_mismatch' });
    expect(bal()).toBe(1000);
    expect(env.DB.q("SELECT reason FROM unmatched_payments WHERE stripe_event_id='evt_live_b'")[0].reason).toBe('livemode_mismatch');
    const inv = await worker.fetch(await signed({ id: 'evt_live_b_inv', type: 'invoice.paid', livemode: false, data: { object: { livemode: true, subscription: 'sub_1' } } }), env);
    expect(inv.status).toBe(200);
    expect(await inv.json()).toMatchObject({ ignored: true, reason: 'livemode_mismatch' });
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).filter((x) => x.msg === 'livemode_mismatch' && x.alert === true)).toHaveLength(2);
  });
  it('rejects wrong secret', async () => { expect((await worker.fetch(await signed(evt(), 'nope'), env)).status).toBe(400); });
});

describe('subscription renewals', () => {
  const inv = (id) => ({ id, type: 'invoice.paid', livemode: false, data: { object: { subscription: 'sub_1', customer: 'cus_1', customer_email: 't@example.com', billing_reason: 'subscription_cycle', amount_paid: 1000, currency: 'usd', payment_intent: `pi_${id}`, charge: `ch_${id}`, lines: { data: [{ quantity: 1, currency: 'usd', price: { id: 'price_test_plan', unit_amount: 1000, currency: 'usd' } }] } } } });
  const sub = (id, type, status) => ({ id, type, livemode: false, data: { object: { id: 'sub_1', customer: 'cus_1', status } } });
  beforeEach(() => { stripeFetch(); env.DB.q("UPDATE accounts SET subscription_id='sub_1', subscription_status='active', stripe_customer_id='cus_1' WHERE id='u1'"); });
  it('invoice.paid tops up (idempotent)', async () => {
    const j = await (await worker.fetch(await signed(inv('evt_r1')), env)).json();
    expect(j.granted).toBe(10000); expect(bal()).toBe(11000);
    await worker.fetch(await signed(inv('evt_r1')), env);
    expect(bal()).toBe(11000);
    await worker.fetch(await signed(inv('evt_r2')), env);
    expect(bal()).toBe(21000);
  });
  it('first invoice (subscription_create) after checkout does not double-grant', async () => {
    env.DB.q("UPDATE accounts SET subscription_id=NULL, subscription_status=NULL, stripe_customer_id=NULL WHERE id='u1'");
    const co = evt({ id: 'evt_co' }); co.data.object.mode = 'subscription'; co.data.object.line_items = li('price_test_plan'); co.data.object.subscription = 'sub_1'; co.data.object.customer = 'cus_1';
    await worker.fetch(await signed(co), env); expect(bal()).toBe(11000);
    const first = inv('evt_first'); first.data.object.billing_reason = 'subscription_create';
    const j = await (await worker.fetch(await signed(first), env)).json();
    expect(j.ignored).toBe('billing_reason_subscription_create'); expect(bal()).toBe(11000);
    await worker.fetch(await signed(inv('evt_cycle')), env); expect(bal()).toBe(21000);
  });
  it('subscription.deleted stops renewals', async () => {
    const j = await (await worker.fetch(await signed(sub('evt_d', 'customer.subscription.deleted')), env)).json();
    expect(j.renewals).toBe('stopped');
    const k = await (await worker.fetch(await signed(inv('evt_r3')), env)).json();
    expect(k.ignored).toBe('subscription_not_active'); expect(bal()).toBe(1000);
  });
  it('subscription.updated past_due stops, active resumes', async () => {
    await worker.fetch(await signed(sub('evt_u1', 'customer.subscription.updated', 'past_due')), env);
    await worker.fetch(await signed(inv('evt_r4')), env); expect(bal()).toBe(1000);
    await worker.fetch(await signed(sub('evt_u2', 'customer.subscription.updated', 'active')), env);
    await worker.fetch(await signed(inv('evt_r5')), env); expect(bal()).toBe(11000);
  });
});

const req = (body = { messages: [{ role: 'user', content: 'hi' }] }, { origin = O, auth = `Bearer ${KEY}` } = {}) =>
  new Request('https://api.rathor.ai/v1/chat', { method: 'POST', body: JSON.stringify(body), headers: { Origin: origin, ...(auth ? { Authorization: auth } : {}) } });

describe('auth', () => {
  it('blocks other origins', async () => { expect((await worker.fetch(req(undefined, { origin: 'https://evil.example' }), env)).status).toBe(403); });
  it('401 without key, wrong key, or revoked key', async () => {
    expect((await worker.fetch(req(undefined, { auth: null }), env)).status).toBe(401);
    expect((await worker.fetch(req(undefined, { auth: `Bearer ${['rth', 'test', 'wrongwrongwrongwrongwrong'].join('_')}` }), env)).status).toBe(401);
    env.DB.q('UPDATE api_keys SET revoked = 1');
    expect((await worker.fetch(req(), env)).status).toBe(401);
  });
  it('passes with valid key; only hashes stored', async () => {
    okFetch();
    expect((await worker.fetch(req(), env)).status).toBe(200);
    const rows = env.DB.q('SELECT key_hash FROM api_keys');
    expect(rows[0].key_hash).toMatch(/^[0-9a-f]{64}$/); expect(rows[0].key_hash).not.toContain(KEY);
  });
  it('constant-time compare', () => { expect(constantTimeEqual('abc', 'abc')).toBe(true); expect(constantTimeEqual('abc', 'abd')).toBe(false); expect(constantTimeEqual('abc', 'ab')).toBe(false); });
});

describe('model allowlist and max_tokens cap', () => {
  it('rejects unlisted model with 400', async () => {
    const f = okFetch();
    const r = await worker.fetch(req({ model: 'some-expensive-model', messages: [] }), env);
    expect(r.status).toBe(400); expect((await r.json()).error).toBe('model_not_allowed'); expect(f).not.toHaveBeenCalled();
  });
  it('rejects max_tokens zero or non-integer; clamps above 4096', async () => {
    for (const mt of [0, 1.5, '100']) {
      const r = await worker.fetch(req({ model: 'grok-4.3', max_tokens: mt, messages: [] }), env);
      expect(r.status).toBe(400);
    }
    const f = okFetch();
    const r = await worker.fetch(req({ model: 'grok-4.3', max_tokens: MAX_TOKENS_CAP + 1, messages: [] }), env);
    expect(r.status).toBe(200);
    expect(JSON.parse(f.mock.calls[0][1].body).max_tokens).toBe(MAX_TOKENS_CAP);
  });
  it('forwards capped max_tokens and allowlisted model only', async () => {
    const f = okFetch();
    await worker.fetch(req({ model: 'grok-4.7', max_tokens: 10, n: 50, tools: [1], messages: [] }), env);
    const sent = JSON.parse(f.mock.calls[0][1].body);
    expect(sent).toEqual({ model: 'grok-4.7', max_tokens: 10, messages: [], stream: false });
  });
});

describe('RATES_VERIFIED gate', () => {
  it('defaults to false', () => { expect(RATES_VERIFIED).toBe(false); });
  it('503 and no upstream call unless RATES_VERIFIED is "true"', async () => {
    const f = okFetch();
    for (const v of [undefined, 'false', '1']) {
      env.RATES_VERIFIED = v;
      expect((await worker.fetch(req(), env)).status).toBe(503);
    }
    expect(f).not.toHaveBeenCalled();
  });
});

describe('hold / settle', () => {
  it('holds max cost, settles actual, releases rest', async () => {
    let heldDuringCall;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => {
      heldDuringCall = env.DB.q("SELECT amount FROM holds WHERE status='open'")[0]?.amount;
      return new Response(JSON.stringify({ choices: [], usage: { prompt_tokens: 1000, completion_tokens: 1000 } }));
    });
    const body = { model: 'grok-4.3', max_tokens: 2000, messages: [{ role: 'user', content: 'hi' }] };
    const j = await (await worker.fetch(req(body), env)).json();
    const expectedHold = holdEstimate('grok-4.3', new TextEncoder().encode(JSON.stringify(body.messages)).length, 1, 2000);
    expect(heldDuringCall).toBe(expectedHold); expect(expectedHold).toBeGreaterThan(6);
    expect(j.rathor.credits_held).toBe(expectedHold);
    expect(j.rathor.credits_charged).toBe(6);
    expect(bal()).toBe(994);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM holds WHERE status='open'")[0].n).toBe(0);
    expect(env.DB.q("SELECT delta FROM ledger WHERE reason='chat'")[0].delta).toBe(-6);
  });
  it('HOLD-HEADROOM-1: usage above the hold and the balance charges the balance, never below 0, and freezes', async () => {
    okFetch({ prompt_tokens: 10_000_000, completion_tokens: 10_000_000 });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const j = await (await worker.fetch(req({ max_tokens: 100, messages: [] }), env)).json();
    expect(j.rathor.credits_charged).toBe(1000); expect(bal()).toBe(0);
    expect(env.DB.q("SELECT settle_locked FROM accounts WHERE id='u1'")[0].settle_locked).toBe(1);
  });
  it('402 when balance cannot cover the hold; upstream not called', async () => {
    env.DB.q("UPDATE accounts SET balance = 1 WHERE id='u1'");
    const f = okFetch();
    expect((await worker.fetch(req({ max_tokens: MAX_TOKENS_CAP, messages: [] }), env)).status).toBe(402);
    expect(f).not.toHaveBeenCalled();
  });
  it('releases the hold on upstream error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('x', { status: 500 }));
    expect((await worker.fetch(req(), env)).status).toBe(502);
    expect(bal()).toBe(1000);
    expect(env.DB.q("SELECT status FROM holds")[0].status).toBe('released');
  });
  it('releases the hold if upstream body is not JSON', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('not json', { status: 200 }));
    expect((await worker.fetch(req(), env)).status).toBe(502);
    expect(env.DB.q('SELECT status FROM holds')[0].status).toBe('released'); expect(bal()).toBe(1000);
  });
  it('releases the hold if settleHold throws', async () => {
    okFetch();
    const orig = env.DB.batch.bind(env.DB);
    env.DB.batch = async (st) => { if (st.some((x) => /'settled'/.test(x.sql))) throw new Error('D1 down'); return orig(st); };
    await expect(worker.fetch(req(), env)).rejects.toThrow('D1 down');
    env.DB.batch = orig;
    expect(env.DB.q('SELECT status FROM holds')[0].status).toBe('released'); expect(bal()).toBe(1000);
  });
  it('concurrent calls cannot double-spend', async () => {
    // Each call holds holdEstimate(grok-4.3, 2 bytes, 0 messages, 2000) = 67 credits; balance 140 covers at most 2.
    env.DB.q("UPDATE accounts SET balance = 140 WHERE id='u1'");
    const hold = holdEstimate('grok-4.3', 2, 0, 2000);
    let inflight = 0, peak = 0;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => {
      inflight++; peak = Math.max(peak, inflight); await new Promise((r) => setTimeout(r, 20)); inflight--;
      return new Response(JSON.stringify({ choices: [], usage: { prompt_tokens: 0, completion_tokens: 2000 } }));
    });
    const rs = await Promise.all(Array.from({ length: 8 }, () => worker.fetch(req({ max_tokens: 2000, messages: [] }), env)));
    const ok = rs.filter((r) => r.status === 200).length;
    expect(hold).toBe(67); expect(ok).toBe(Math.floor(140 / hold));
    expect(rs.filter((r) => r.status === 402).length).toBe(8 - ok);
    expect(peak).toBe(ok);
    expect(bal()).toBeGreaterThanOrEqual(0);
  });
  it('min 1 credit (0.1¢)', () => { expect(creditsForUsage('grok-4.3', {})).toBe(1); });
});

describe('Mercy-Checks (launch rules, MERCY-LAUNCH-1)', () => {
  it('is no longer the baseline deny-list', () => { expect(MERCY_CHECKS_BASELINE).toBe(false); });
  it('denies request before upstream call, no hold taken', async () => {
    const f = okFetch();
    const r = await worker.fetch(req({ messages: [{ role: 'user', content: 'Explain HOW TO MAKE A  BOMB' }] }), env);
    const j = await r.json();
    expect(r.status).toBe(400); expect(j.stage).toBe('request'); expect(j.reason).toBe('explosives_weaponisation');
    expect(f).not.toHaveBeenCalled(); expect(env.DB.q('SELECT COUNT(*) AS n FROM holds')[0].n).toBe(0);
  });
  it('checks output and withholds denied content', async () => {
    okFetch(undefined, 'step one: how to make a bomb');
    const r = await worker.fetch(req(), env); const j = await r.json();
    expect(r.status).toBe(422); expect(j.stage).toBe('response'); expect(j.reason).toBe('explosives_weaponisation'); expect(JSON.stringify(j)).not.toContain('step one');
  });
  it('allows ordinary text', () => { expect(mercyChecks.checkRequest({ messages: [{ role: 'user', content: 'hello' }] }).allowed).toBe(true); });
});

describe('PRICING-1 launch pricing', () => {
  it('45% markup, 1 credit = $0.001, $1.00 trial', () => {
    expect(MARKUP).toBe(1.45); expect(USD_PER_CREDIT).toBe(0.001);
    expect(PRICING.trialCredits).toBe(1000); expect(creditsToUsd(PRICING.trialCredits)).toBe(1);
  });
  it('base xAI rates stored un-marked-up', () => {
    expect(XAI_RATES_USD_PER_MTOK['grok-4.7']).toEqual({ input: 2.0, cachedInput: 0.5, output: 6.0 });
    expect(XAI_RATES_USD_PER_MTOK['grok-4.3']).toEqual({ input: 1.25, cachedInput: 0.2, output: 2.5 });
  });
  it('rounds UP at the 0.1 cent unit', () => {
    // grok-4.3, 1 output token: $0.0000025 * 1.45 = 0.003625 credits -> 1
    expect(creditsForTokens('grok-4.3', 0, 1)).toBe(1);
    // grok-4.7, 1000 output: $0.006 * 1.45 = $0.0087 = 8.7 credits -> 9
    expect(creditsForTokens('grok-4.7', 0, 1000)).toBe(9);
    // exact boundary (no float drift): 10,000 out = $0.06*1.45 = 87 credits exactly
    expect(creditsForTokens('grok-4.7', 0, 10000)).toBe(87);
    expect(creditsForTokens('grok-4.7', 0, 10001)).toBe(88);
  });
  it('sample grok-4.7 request at 45%: 1000 in + 500 out = 8 credits ($0.008)', () => {
    // ($2*1000 + $6*500)/1e6 = $0.005; *1.45 = $0.00725 = 7.25 credits -> 8
    expect(creditsForUsage('grok-4.7', { prompt_tokens: 1000, completion_tokens: 500 })).toBe(8);
    expect(creditsToUsd(8)).toBe(0.008);
  });
  it('sample grok-4.3 request at 45%: 1000 in + 500 out = 4 credits ($0.004)', () => {
    // ($1.25*1000 + $2.5*500)/1e6 = $0.0025; *1.45 = $0.003625 = 3.625 credits -> 4
    expect(creditsForUsage('grok-4.3', { prompt_tokens: 1000, completion_tokens: 500 })).toBe(4);
  });
  it('uses the cached-input rate for cached prompt tokens', () => {
    // grok-4.7: 10,000 cached in = $0.005 *1.45 = 7.25 -> 8 (vs 29 uncached)
    expect(creditsForUsage('grok-4.7', { prompt_tokens: 10000, completion_tokens: 0, prompt_tokens_details: { cached_tokens: 10000 } })).toBe(8);
    expect(creditsForUsage('grok-4.7', { prompt_tokens: 10000, completion_tokens: 0 })).toBe(29);
  });
  it('allowlist is exactly grok-4.7 and grok-4.3', () => {
    expect([...ALLOWED_MODELS]).toEqual(['grok-4.7', 'grok-4.3']);
    expect(ALLOWED_MODELS).toContain(DEFAULT_MODEL);
    for (const m of ALLOWED_MODELS) expect(XAI_RATES_USD_PER_MTOK[m]).toBeDefined();
  });
  it('rejects a non-allowlisted model (old placeholder ids too)', async () => {
    const f = okFetch();
    for (const model of ['grok-4', 'grok-3-mini', 'grok-4.7-mini']) {
      const r = await worker.fetch(req({ model, messages: [] }), env);
      expect(r.status).toBe(400); expect((await r.json()).error).toBe('model_not_allowed');
    }
    expect(f).not.toHaveBeenCalled();
  });
  it('charges grok-4.7 usage end to end', async () => {
    okFetch({ prompt_tokens: 1000, completion_tokens: 500 });
    const j = await (await worker.fetch(req({ model: 'grok-4.7', messages: [] }), env)).json();
    expect(j.rathor.credits_charged).toBe(8); expect(bal()).toBe(992);
  });
  it('smallest pack is $10; nothing smaller is offered', () => {
    expect(MIN_PACK_USD).toBe(10);
    expect(Math.min(...CREDIT_PACKS.map((p) => p.usd))).toBe(10);
    expect(CREDIT_PACKS[0].credits).toBe(10000);
    expect(creditsToUsd(PRICING.monthlyPlanCredits)).toBeGreaterThanOrEqual(10);
  });
  it('webhook refuses a grant smaller than the $10 pack (price, not metadata or amount)', async () => {
    stripeFetch();
    const e = evt({ id: 'evt_small' }); Object.assign(e.data.object, { line_items: li('price_test_small', { unit: 500 }), amount_total: 500 }); e.data.object.metadata.credits = '5000';
    const r = await worker.fetch(await signed(e), env); const j = await r.json();
    expect(r.status).toBe(200); expect(j).toMatchObject({ ignored: true, reason: 'unknown_price' }); expect(bal()).toBe(1000);
    const ok = evt({ id: 'evt_25' }); Object.assign(ok.data.object, { line_items: li('price_test_p25'), amount_total: 2500 });
    expect((await (await worker.fetch(await signed(ok), env)).json()).balance).toBe(26000);
  });
  it('still charges when Mercy-Checks withholds the reply (tokens were used)', async () => {
    okFetch({ prompt_tokens: 1000, completion_tokens: 500 }, 'step one: how to make a bomb');
    const r = await worker.fetch(req({ model: 'grok-4.7', messages: [] }), env); const j = await r.json();
    expect(r.status).toBe(422); expect(j.rathor.credits_charged).toBe(8); expect(bal()).toBe(992);
    expect(env.DB.q("SELECT delta FROM ledger WHERE reason='chat'")[0].delta).toBe(-8);
  });
});

describe('PRICING-2/3: webhook grants by line-item price id', () => {
  const unmatched = () => env.DB.q('SELECT * FROM unmatched_payments');
  let f;
  beforeEach(() => { f = stripeFetch(); });

  it('fetches line items server-side with the secret key and grants each pack and the trial', async () => {
    for (const [i, [pid, cr]] of [['price_test_p10', 10000], ['price_test_p25', 25000], ['price_test_p50', 50000], ['price_test_trial', 1000]].entries()) {
      const e = evt({ id: `evt_pid_${i}` }); e.data.object.line_items = li(pid);
      expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ granted: cr, matched_by: 'price_id' });
    }
    expect(bal()).toBe(1000 + 10000 + 25000 + 50000 + 1000);
    const call = f.mock.calls.find((c) => String(c[0]).includes('api.stripe.com'));
    expect(String(call[0])).toBe('https://api.stripe.com/v1/checkout/sessions/cs_test_evt_pid_0/line_items?limit=10');
    expect(call[1].headers.Authorization).toBe('Bearer sk_test_FAKE_not_a_real_key');
    expect(unmatched()).toHaveLength(0);
  });
  it('coupon-discounted pack grants the full pack credits (promos discount money, not credits)', async () => {
    const e = evt({ id: 'evt_coupon' }); Object.assign(e.data.object, { line_items: li('price_test_p25', { discount: 500 }), amount_total: 2000, total_details: { amount_discount: 500 } });
    const r = await worker.fetch(await signed(e), env);
    expect(r.status).toBe(200); expect(await r.json()).toMatchObject({ granted: 25000, kind: 'pack' }); expect(bal()).toBe(26000);
  });
  it('PRICING-4: a 100%-off checkout grants nothing (no_payment_required, or amount_total 0)', async () => {
    const free = evt({ id: 'evt_coupon100' }); Object.assign(free.data.object, { line_items: li('price_test_p10', { discount: 1000 }), amount_total: 0, payment_status: 'no_payment_required' });
    let r = await worker.fetch(await signed(free), env);
    expect(r.status).toBe(200); expect(await r.json()).toMatchObject({ ignored: true, reason: 'payment_status_no_payment_required' });
    const zero = evt({ id: 'evt_coupon100b' }); Object.assign(zero.data.object, { line_items: li('price_test_p10', { discount: 1000 }), amount_total: 0 });
    r = await worker.fetch(await signed(zero), env);
    expect(r.status).toBe(200); expect(await r.json()).toMatchObject({ ignored: true, reason: 'zero_amount_paid' });
    const trial = evt({ id: 'evt_coupon100t' }); Object.assign(trial.data.object, { line_items: li('price_test_trial', { discount: 100 }), amount_total: 0, payment_status: 'no_payment_required' });
    expect(await (await worker.fetch(await signed(trial), env)).json()).toMatchObject({ ignored: true });
    expect(bal()).toBe(1000); expect(env.DB.q('SELECT COUNT(*) AS n FROM trial_redemptions')[0].n).toBe(0);
    expect(unmatched().map((u) => u.reason)).toEqual(['payment_status_no_payment_required', 'zero_amount_paid', 'payment_status_no_payment_required']);
    expect(f.mock.calls.filter((c) => String(c[0]).includes('api.stripe.com'))).toHaveLength(0);
  });
  it('taxed pack (Stripe Tax adds to amount_total) still grants', async () => {
    const e = evt({ id: 'evt_tax' }); Object.assign(e.data.object, { line_items: li('price_test_p10', { tax: 130 }), amount_total: 1130, total_details: { amount_tax: 130 } });
    expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ granted: 10000 }); expect(bal()).toBe(11000);
  });
  it('unit amount mismatch on a known price: 200 ignored, recorded, no grant', async () => {
    const e = evt({ id: 'evt_amt' }); e.data.object.line_items = li('price_test_p50', { unit: 1000 });
    const r = await worker.fetch(await signed(e), env);
    expect(r.status).toBe(200); expect(await r.json()).toMatchObject({ ignored: true, reason: 'amount_mismatch' }); expect(bal()).toBe(1000);
    expect(unmatched()[0]).toMatchObject({ stripe_event_id: 'evt_amt', reason: 'amount_mismatch', price_id: 'price_test_p50', currency: 'usd' });
  });
  it('quantity other than 1 is refused', async () => {
    const e = evt({ id: 'evt_qty' }); e.data.object.line_items = li('price_test_p10', { qty: 3 });
    expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ ignored: true, reason: 'quantity_not_1' }); expect(bal()).toBe(1000);
  });
  it('multiple line items, or none, are refused', async () => {
    const two = li('price_test_p10'); two.data.push(li('price_test_p25').data[0]);
    const a = evt({ id: 'evt_two' }); a.data.object.line_items = two;
    expect(await (await worker.fetch(await signed(a), env)).json()).toMatchObject({ ignored: true, reason: 'multiple_line_items' });
    const b = evt({ id: 'evt_none' }); b.data.object.line_items = { object: 'list', data: [], has_more: false };
    expect(await (await worker.fetch(await signed(b), env)).json()).toMatchObject({ ignored: true, reason: 'no_line_items' });
    expect(bal()).toBe(1000);
  });
  it('currency mismatch: 200 ignored, recorded, no grant', async () => {
    const e = evt({ id: 'evt_cur1' }); Object.assign(e.data.object, { line_items: li('price_test_p10', { currency: 'eur' }), currency: 'eur' });
    const r = await worker.fetch(await signed(e), env);
    expect(r.status).toBe(200); expect(await r.json()).toMatchObject({ ignored: true, reason: 'currency_mismatch' });
    expect(bal()).toBe(1000); expect(unmatched()[0].currency).toBe('eur');
  });
  it('unknown price id (also when price vars are unset): recorded and answered 200', async () => {
    const e = evt({ id: 'evt_unk' }); e.data.object.line_items = li('price_someone_else');
    expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ ignored: true, reason: 'unknown_price' });
    for (const k of Object.keys(PRICES)) env[k] = '';
    const e2 = evt({ id: 'evt_unk2' });
    expect(await (await worker.fetch(await signed(e2), env)).json()).toMatchObject({ ignored: true, reason: 'unknown_price' });
    expect(bal()).toBe(1000); expect(unmatched().map((u) => u.price_id)).toEqual(['price_someone_else', 'price_test_p10']);
  });
  it('no amount-only fallback: a $10 checkout whose line item has no price id is refused', async () => {
    const l = li('x'); delete l.data[0].price;
    const e = evt({ id: 'evt_fb' }); e.data.object.line_items = l;
    expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ ignored: true, reason: 'price_missing' }); expect(bal()).toBe(1000);
  });
  it('PRICING-4: transient line-items failures (429, 5xx, network, bad JSON) answer 502, log stripe_fetch_transient, record nothing; a retry grants', async () => {
    f.mockRestore();
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const e = evt({ id: 'evt_flaky' });
    const cases = [() => new Response('{"error":{}}', { status: 429 }), () => new Response('{"error":{}}', { status: 500 }), () => new Response('{"error":{}}', { status: 503 }), () => { throw new Error('network'); }, () => new Response('not json', { status: 200 })];
    for (const c of cases) {
      vi.mocked(globalThis.fetch).mockRestore?.(); stripeFetch(async () => c());
      const r = await worker.fetch(await signed(e), env);
      expect(r.status).toBe(502); expect(await r.json()).toMatchObject({ error: 'stripe_fetch_transient', retry: true });
    }
    expect(bal()).toBe(1000); expect(unmatched()).toHaveLength(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_flaky'")[0].n).toBe(0);
    const logs = err.mock.calls.map((c) => JSON.parse(c[0]));
    expect(logs.filter((x) => x.msg === 'stripe_fetch_transient').map((x) => x.status)).toEqual([429, 500, 503, null, 200]);
    expect(logs.filter((x) => x.msg === 'stripe_fetch_transient').every((x) => x.alert === true)).toBe(true);
    expect(logs.filter((x) => x.msg === 'stripe_webhook_retry')).toHaveLength(5);
    expect(logs.find((x) => x.msg === 'stripe_webhook_retry')).toMatchObject({ status: 502, error: 'stripe_fetch_transient', event_id: 'evt_flaky', type: 'checkout.session.completed' });
    vi.mocked(globalThis.fetch).mockRestore?.(); stripeFetch(async () => null);
    expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ granted: 10000 });
  });
  it('PRICING-4: 401/403 from Stripe answers 500 with a stripe_auth_error alert, records nothing', async () => {
    f.mockRestore();
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    for (const status of [401, 403]) {
      vi.mocked(globalThis.fetch).mockRestore?.(); stripeFetch(async () => new Response('{"error":{"type":"invalid_request_error"}}', { status }));
      const r = await worker.fetch(await signed(evt({ id: `evt_auth_${status}` })), env);
      expect(r.status).toBe(500); expect(await r.json()).toMatchObject({ error: 'stripe_auth_error', retry: true, status });
    }
    const logs = err.mock.calls.map((c) => JSON.parse(c[0]));
    expect(logs.filter((x) => x.msg === 'stripe_auth_error')).toMatchObject([{ alert: true, status: 401 }, { alert: true, status: 403 }]);
    expect(logs.filter((x) => x.msg === 'stripe_webhook_retry').map((x) => x.status)).toEqual([500, 500]);
    expect(bal()).toBe(1000); expect(unmatched()).toHaveLength(0); expect(env.DB.q('SELECT COUNT(*) AS n FROM stripe_events')[0].n).toBe(0);
  });
  it('STRIPE-404-RETRY-1: line-items fetch 404 answers 5xx, 0 credits, no processed-event row, no unmatched row', async () => {
    const e = evt({ id: 'evt_404' }); delete e.data.object.line_items; // the mock answers 404 for unknown sessions
    const r = await worker.fetch(await signed(e), env);
    expect(r.status).toBeGreaterThanOrEqual(500); expect(r.status).toBeLessThan(600);
    expect(await r.json()).toMatchObject({ error: 'stripe_fetch_transient', retry: true, status: 404 });
    expect(bal()).toBe(1000);
    expect(unmatched()).toHaveLength(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_404'")[0].n).toBe(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE stripe_event_id='evt_404'")[0].n).toBe(0);
  });
  it('STRIPE-404-RETRY-1: network throw answers 5xx, 0 credits, no processed-event row, no unmatched row', async () => {
    f.mockRestore();
    stripeFetch(async () => { throw new Error('network'); });
    const e = evt({ id: 'evt_net' });
    const r = await worker.fetch(await signed(e), env);
    expect(r.status).toBeGreaterThanOrEqual(500); expect(r.status).toBeLessThan(600);
    expect(await r.json()).toMatchObject({ error: 'stripe_fetch_transient', retry: true });
    expect(bal()).toBe(1000); expect(unmatched()).toHaveLength(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_net'")[0].n).toBe(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE stripe_event_id='evt_net'")[0].n).toBe(0);
  });
  it('STRIPE-404-RETRY-1: 500 from Stripe answers 5xx, 0 credits, no processed-event row, no unmatched row', async () => {
    f.mockRestore();
    stripeFetch(async () => new Response('{"error":{}}', { status: 500 }));
    const e = evt({ id: 'evt_500' });
    const r = await worker.fetch(await signed(e), env);
    expect(r.status).toBeGreaterThanOrEqual(500); expect(r.status).toBeLessThan(600);
    expect(await r.json()).toMatchObject({ error: 'stripe_fetch_transient', retry: true, status: 500 });
    expect(bal()).toBe(1000); expect(unmatched()).toHaveLength(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_500'")[0].n).toBe(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE stripe_event_id='evt_500'")[0].n).toBe(0);
  });
  it('STRIPE-404-RETRY-1: same event retried after a failed fetch grants credits exactly once', async () => {
    f.mockRestore();
    const e = evt({ id: 'evt_retry1' });
    stripeFetch(async () => new Response('{"error":{"type":"invalid_request_error"}}', { status: 404 }));
    const fail = await worker.fetch(await signed(e), env);
    expect(fail.status).toBeGreaterThanOrEqual(500); expect(fail.status).toBeLessThan(600);
    expect(await fail.json()).toMatchObject({ error: 'stripe_fetch_transient', retry: true });
    expect(bal()).toBe(1000);
    expect(unmatched()).toHaveLength(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_retry1'")[0].n).toBe(0);
    vi.mocked(globalThis.fetch).mockRestore?.();
    stripeFetch();
    const ok = await worker.fetch(await signed(e), env);
    expect(ok.status).toBe(200);
    expect(await ok.json()).toMatchObject({ granted: 10000, matched_by: 'price_id' });
    expect(bal()).toBe(11000);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_retry1'")[0].n).toBe(1);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE stripe_event_id='evt_retry1' AND delta=10000")[0].n).toBe(1);
    expect(unmatched()).toHaveLength(0);
    expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ duplicate: true });
    expect(bal()).toBe(11000);
  });
  it('STRIPE-404-RETRY-1 exhaust: old 404 writes one stripe_fetch_exhausted row, 200 ignored, 0 credits, no stripe_events; replay stays one row', async () => {
    const NOW_SEC = 1_700_000_000;
    vi.spyOn(Date, 'now').mockReturnValue(NOW_SEC * 1000);
    env.STRIPE_RETRY_EXHAUST_SECONDS = '21600';
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const e = evt({ id: 'evt_old404', created: NOW_SEC - 21600 });
    delete e.data.object.line_items;
    const r = await worker.fetch(await signed(e), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'stripe_fetch_exhausted' });
    expect(bal()).toBe(1000);
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_old404', reason: 'stripe_fetch_exhausted', detail: '404', mode: 'payment' }]);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_old404'")[0].n).toBe(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE stripe_event_id='evt_old404'")[0].n).toBe(0);
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).filter((x) => x.msg === 'stripe_fetch_exhausted')).toMatchObject([{ alert: true, status: '404', event_id: 'evt_old404' }]);
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).find((x) => x.msg === 'stripe_fetch_transient')).toMatchObject({ alert: true, status: 404 });
    const again = await worker.fetch(await signed(e), env);
    expect(again.status).toBe(200);
    expect(await again.json()).toMatchObject({ ignored: true, reason: 'stripe_fetch_exhausted' });
    expect(unmatched()).toHaveLength(1);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_old404'")[0].n).toBe(0);
  });
  it('STRIPE-404-RETRY-1 exhaust: fresh 404 (and missing/invalid created) answers 502 with nothing written', async () => {
    const NOW_SEC = 1_700_000_000;
    vi.spyOn(Date, 'now').mockReturnValue(NOW_SEC * 1000);
    env.STRIPE_RETRY_EXHAUST_SECONDS = '21600';
    const fresh = evt({ id: 'evt_fresh404', created: NOW_SEC - 21599 });
    delete fresh.data.object.line_items;
    const r = await worker.fetch(await signed(fresh), env);
    expect(r.status).toBe(502);
    expect(await r.json()).toMatchObject({ error: 'stripe_fetch_transient', retry: true, status: 404 });
    const missing = evt({ id: 'evt_nocreated' });
    delete missing.data.object.line_items;
    expect((await worker.fetch(await signed(missing), env)).status).toBe(502);
    const bad = evt({ id: 'evt_badcreated', created: 'not-an-int' });
    delete bad.data.object.line_items;
    expect((await worker.fetch(await signed(bad), env)).status).toBe(502);
    expect(unmatched()).toHaveLength(0);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM stripe_events')[0].n).toBe(0);
    expect(bal()).toBe(1000);
  });
  it('STRIPE-404-RETRY-1 exhaust: manual resend after exhaustion with a good fetch grants exactly once', async () => {
    const NOW_SEC = 1_700_000_000;
    vi.spyOn(Date, 'now').mockReturnValue(NOW_SEC * 1000);
    env.STRIPE_RETRY_EXHAUST_SECONDS = '21600';
    f.mockRestore();
    stripeFetch(async () => new Response('{"error":{"type":"invalid_request_error"}}', { status: 404 }));
    const e = evt({ id: 'evt_resend', created: NOW_SEC - 21600 });
    expect((await worker.fetch(await signed(e), env)).status).toBe(200);
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_resend', reason: 'stripe_fetch_exhausted' }]);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_resend'")[0].n).toBe(0);
    expect(bal()).toBe(1000);
    vi.mocked(globalThis.fetch).mockRestore?.();
    stripeFetch();
    const ok = await worker.fetch(await signed(e), env);
    expect(ok.status).toBe(200);
    expect(await ok.json()).toMatchObject({ granted: 10000, matched_by: 'price_id' });
    expect(bal()).toBe(11000);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_resend'")[0].n).toBe(1);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE stripe_event_id='evt_resend' AND delta=10000")[0].n).toBe(1);
    expect(unmatched()).toHaveLength(1);
    expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ duplicate: true });
    expect(bal()).toBe(11000);
  });
  it('STRIPE_RETRY_EXHAUST_SECONDS defaults to 262800 when unset, invalid, or under 3600', () => {
    expect(STRIPE_RETRY_EXHAUST_SECONDS_DEFAULT).toBe(262800);
    expect(stripeRetryExhaustSeconds({})).toBe(262800);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: 'nope' })).toBe(262800);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: '21600' })).toBe(21600);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: '252000' })).toBe(252000);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: '0' })).toBe(262800);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: '3599' })).toBe(262800);
    expect(stripeRetryExhaustSeconds({ STRIPE_RETRY_EXHAUST_SECONDS: '3600' })).toBe(3600);
  });
  it('LIVE-GATE-1 (c): LIVE_MODE true + test key, and LIVE_MODE false + live key, both 500 before any Stripe fetch', async () => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    env.LIVE_MODE = 'true'; env.STRIPE_SECRET_KEY = SK_TEST;
    let r = await worker.fetch(await signed(evt({ id: 'evt_mm_live_test' })), env);
    expect(r.status).toBe(500); expect(await r.json()).toMatchObject({ error: 'stripe_key_mode_mismatch' });
    env.LIVE_MODE = 'false'; env.STRIPE_SECRET_KEY = SK_LIVE;
    r = await worker.fetch(await signed(evt({ id: 'evt_mm_sand_live' })), env);
    expect(r.status).toBe(500); expect(await r.json()).toMatchObject({ error: 'stripe_key_mode_mismatch' });
    env.LIVE_MODE = 'false'; env.STRIPE_SECRET_KEY = RK_LIVE;
    r = await worker.fetch(await signed(evt({ id: 'evt_mm_sand_rk' })), env);
    expect(r.status).toBe(500);
    expect(f.mock.calls.filter((c) => String(c[0]).includes('api.stripe.com'))).toHaveLength(0);
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).filter((x) => x.msg === 'stripe_key_mode_mismatch' && x.alert === true)).toHaveLength(3);
    expect(bal()).toBe(1000); expect(unmatched()).toHaveLength(0);
    env.LIVE_MODE = 'false'; env.STRIPE_SECRET_KEY = RK_TEST;
    expect(await (await worker.fetch(await signed(evt({ id: 'evt_key_rk' })), env)).json()).toMatchObject({ granted: 10000 });
  });
  it('missing STRIPE_SECRET_KEY answers 500 (config error, Stripe retries), no grant', async () => {
    delete env.STRIPE_SECRET_KEY;
    const r = await worker.fetch(await signed(evt({ id: 'evt_nokey' })), env);
    expect(r.status).toBe(500); expect(await r.json()).toMatchObject({ error: 'stripe_secret_key_missing' }); expect(bal()).toBe(1000);
  });
  it('trial by amount ($1 with a non-trial price id) is refused', async () => {
    const e = evt({ id: 'evt_trial_amt' }); Object.assign(e.data.object, { line_items: li('price_other_1usd', { unit: 100 }), amount_total: 100 });
    expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ ignored: true, reason: 'unknown_price' });
    expect(bal()).toBe(1000); expect(env.DB.q('SELECT COUNT(*) AS n FROM trial_redemptions')[0].n).toBe(0);
  });
  it('trial at most once per account: second trial recorded and answered 200 ignored; other accounts unaffected', async () => {
    const t1 = evt({ id: 'evt_t1' }); t1.data.object.line_items = li('price_test_trial');
    expect(await (await worker.fetch(await signed(t1), env)).json()).toMatchObject({ granted: 1000, kind: 'trial' });
    const t2 = evt({ id: 'evt_t2' }); t2.data.object.line_items = li('price_test_trial');
    const r = await worker.fetch(await signed(t2), env);
    expect(r.status).toBe(200); expect(await r.json()).toMatchObject({ ignored: true, reason: 'trial_already_used' });
    expect(bal()).toBe(2000);
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_t2', reason: 'trial_already_used', account_id: 'u1', price_id: 'price_test_trial' }]);
    // replay of the refused event stays a duplicate; replay of the granted one too
    expect(await (await worker.fetch(await signed(t2), env)).json()).toMatchObject({ duplicate: true });
    expect(await (await worker.fetch(await signed(t1), env)).json()).toMatchObject({ duplicate: true });
    env.DB.q("INSERT INTO accounts (id, balance) VALUES ('u2', 0)");
    const t3 = evt({ id: 'evt_t3' }); Object.assign(t3.data.object, { client_reference_id: 'u2', line_items: li('price_test_trial') });
    expect(await (await worker.fetch(await signed(t3), env)).json()).toMatchObject({ granted: 1000 });
    expect(bal()).toBe(2000); expect(unmatched()).toHaveLength(1);
  });
  it('concurrent trial checkouts for one account grant once', async () => {
    await Promise.all(['a', 'b', 'c'].map(async (x) => { const e = evt({ id: `evt_tc_${x}` }); e.data.object.line_items = li('price_test_trial'); return worker.fetch(await signed(e), env); }));
    expect(bal()).toBe(2000); expect(unmatched()).toHaveLength(2);
  });
  it('unknown or missing user: recorded and answered 200 ignored (not 400), no account created', async () => {
    const u = evt({ id: 'evt_nouser' }); u.data.object.client_reference_id = 'ghost';
    const r = await worker.fetch(await signed(u), env);
    expect(r.status).toBe(200); expect(await r.json()).toMatchObject({ ignored: true, reason: 'unknown_user' });
    expect(env.DB.q("SELECT COUNT(*) AS n FROM accounts WHERE id='ghost'")[0].n).toBe(0);
    const m = evt({ id: 'evt_missing' }); delete m.data.object.client_reference_id;
    const r2 = await worker.fetch(await signed(m), env);
    expect(r2.status).toBe(200); expect(await r2.json()).toMatchObject({ ignored: true, reason: 'no_user' });
    const inv = { id: 'evt_inv_nouser', type: 'invoice.paid', livemode: false, data: { object: { subscription: 'sub_x', billing_reason: 'subscription_cycle', amount_paid: 1000, currency: 'usd' } } };
    const r3 = await worker.fetch(await signed(inv), env);
    expect(r3.status).toBe(200); expect(await r3.json()).toMatchObject({ ignored: true, reason: 'no_user' });
    expect(unmatched().map((x) => x.reason)).toEqual(['unknown_user', 'no_user', 'no_user']);
  });
  it('schema missing (migration 0002/0003 not applied): structured schema_missing log and 500', async () => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    env.DB.q('DROP TABLE unmatched_payments');
    const e = evt({ id: 'evt_s1' }); e.data.object.line_items = li('price_nope');
    const r = await worker.fetch(await signed(e), env);
    expect(r.status).toBe(500); expect(await r.json()).toEqual({ error: 'schema_missing' });
    env.DB.q('DROP TABLE trial_redemptions');
    const t = evt({ id: 'evt_s2' }); t.data.object.line_items = li('price_test_trial');
    expect((await worker.fetch(await signed(t), env)).status).toBe(500);
    expect(bal()).toBe(1000);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events")[0].n).toBe(0);
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).filter((x) => x.msg === 'schema_missing')).toHaveLength(2);
  });
  it('SPROOF-FIX-README-PERMS: migration 0006 missing (no such table: early_disputes) is 500 schema_missing, nothing granted', async () => {
    expect(isSchemaMissing(new Error('D1_ERROR: no such table: early_disputes: SQLITE_ERROR'))).toBe(true);
    expect(isSchemaMissing('no such table: early_disputes')).toBe(true);
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    env.DB.q('DROP TABLE early_disputes');
    const before = bal();
    const r = await worker.fetch(await signed(evt({ id: 'evt_no_0006' })), env);
    expect(r.status).toBe(500); expect(await r.json()).toEqual({ error: 'schema_missing' });
    expect(bal()).toBe(before);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events")[0].n).toBe(0);
    const logs = err.mock.calls.map((c) => { try { return JSON.parse(c[0]); } catch { return {}; } });
    expect(logs.filter((x) => x.msg === 'schema_missing' && /early_disputes/.test(x.error))).toHaveLength(1);
  });
  it('plan vs pack: the plan price cannot be redeemed as a one-time pack, nor a pack as the plan', async () => {
    const planAsPack = evt({ id: 'evt_pp1' }); planAsPack.data.object.line_items = li('price_test_plan');
    expect(await (await worker.fetch(await signed(planAsPack), env)).json()).toMatchObject({ ignored: true, reason: 'price_kind_plan_not_allowed_for_checkout_payment' });
    const packAsPlan = evt({ id: 'evt_pp2' }); Object.assign(packAsPlan.data.object, { mode: 'subscription', line_items: li('price_test_p10') });
    expect(await (await worker.fetch(await signed(packAsPlan), env)).json()).toMatchObject({ ignored: true, reason: 'price_kind_pack_not_allowed_for_checkout_subscription' });
    env.DB.q("UPDATE accounts SET subscription_id='sub_1', subscription_status='active' WHERE id='u1'");
    const inv = { id: 'evt_pp3', type: 'invoice.paid', livemode: false, data: { object: { subscription: 'sub_1', billing_reason: 'subscription_cycle', amount_paid: 1000, currency: 'usd', lines: { data: [{ quantity: 1, pricing: { price_details: { price: 'price_test_p10' }, unit_amount_decimal: '1000' } }] } } } };
    expect(await (await worker.fetch(await signed(inv), env)).json()).toMatchObject({ ignored: true, reason: 'price_kind_pack_not_allowed_for_invoice_cycle' });
    expect(bal()).toBe(1000); expect(unmatched()).toHaveLength(3);
    const plan = evt({ id: 'evt_pp4' }); Object.assign(plan.data.object, { mode: 'subscription', line_items: li('price_test_plan'), subscription: 'sub_1' });
    expect(await (await worker.fetch(await signed(plan), env)).json()).toMatchObject({ granted: 10000, kind: 'plan', matched_by: 'price_id' });
    const inv2 = { ...inv, id: 'evt_pp5' }; inv2.data = { object: { ...inv.data.object, payment_intent: 'pi_evt_pp5', charge: 'ch_evt_pp5', lines: { data: [{ quantity: 1, pricing: { price_details: { price: 'price_test_plan' }, unit_amount_decimal: '1000' } }] } } };
    expect(await (await worker.fetch(await signed(inv2), env)).json()).toMatchObject({ granted: 10000, kind: 'plan', matched_by: 'price_id' });
    expect(bal()).toBe(21000);
  });
  it('renewal with a coupon and tax (amount_paid differs) grants by the invoice line price id', async () => {
    env.DB.q("UPDATE accounts SET subscription_id='sub_1', subscription_status='active' WHERE id='u1'");
    const inv = { id: 'evt_rc', type: 'invoice.paid', livemode: false, data: { object: { subscription: 'sub_1', billing_reason: 'subscription_cycle', amount_paid: 565, currency: 'usd', payment_intent: 'pi_evt_rc', charge: 'ch_evt_rc', lines: { data: [{ quantity: 1, currency: 'usd', price: { id: 'price_test_plan', unit_amount: 1000, currency: 'usd' } }] } } } };
    expect(await (await worker.fetch(await signed(inv), env)).json()).toMatchObject({ granted: 10000 });
  });
  it('PRICING-4: renewal with amount_paid <= 0 (100%-off coupon) is refused, recorded, 200 ignored', async () => {
    env.DB.q("UPDATE accounts SET subscription_id='sub_1', subscription_status='active', stripe_customer_id='cus_1' WHERE id='u1'");
    for (const [id, amt] of [['evt_r0', 0], ['evt_rneg', -5]]) {
      const inv = { id, type: 'invoice.paid', livemode: false, data: { object: { subscription: 'sub_1', customer: 'cus_1', billing_reason: 'subscription_cycle', amount_paid: amt, currency: 'usd', lines: { data: [{ quantity: 1, currency: 'usd', price: { id: 'price_test_plan', unit_amount: 1000, currency: 'usd' } }] } } } };
      const r = await worker.fetch(await signed(inv), env);
      expect(r.status).toBe(200); expect(await r.json()).toMatchObject({ ignored: true, reason: 'zero_amount_paid' });
    }
    expect(bal()).toBe(1000); expect(unmatched().map((u) => [u.reason, u.amount])).toEqual([['zero_amount_paid', 0], ['zero_amount_paid', -5]]);
  });
  it('PRICING-4: no free signup credits: a new account starts at 0 and cannot chat until it pays', async () => {
    const { ensureAccount, SIGNUP_GRANT_CREDITS } = await import('../src/db.js');
    expect(SIGNUP_GRANT_CREDITS).toBe(0);
    await ensureAccount(env.DB, 'u_new'); await ensureAccount(env.DB, 'u_new');
    expect(bal('u_new')).toBe(0); expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE account_id='u_new'")[0].n).toBe(0);
    await ensureAccount(env.DB, 'u1'); expect(bal()).toBe(1000);
    const K2 = ['rth', 'test', 'newaccountkey0123456789'].join('_');
    env.DB.q('INSERT INTO api_keys (key_hash, account_id) VALUES (?1, ?2)', await sha256Hex(K2), 'u_new');
    f.mockRestore(); okFetch();
    const r = await worker.fetch(new Request('https://api.rathor.ai/v1/chat', { method: 'POST', headers: { Origin: O, Authorization: `Bearer ${K2}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }) }), env);
    expect(r.status).toBe(402);
    vi.restoreAllMocks(); stripeFetch();
    const t = evt({ id: 'evt_new_trial' }); Object.assign(t.data.object, { client_reference_id: 'u_new', line_items: li('price_test_trial') });
    expect(await (await worker.fetch(await signed(t), env)).json()).toMatchObject({ granted: 1000, kind: 'trial' });
    expect(bal('u_new')).toBe(1000);
  });
  it('PRICING-4: every non-2xx webhook answer logs stripe_webhook_retry', async () => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    await worker.fetch(await signed(evt({ id: 'evt_badsig' }), 'nope'), env);
    delete env.STRIPE_SECRET_KEY; await worker.fetch(await signed(evt({ id: 'evt_nokey2' })), env);
    const logs = err.mock.calls.map((c) => JSON.parse(c[0])).filter((x) => x.msg === 'stripe_webhook_retry');
    expect(logs.map((x) => [x.status, x.error])).toEqual([[400, 'bad_signature'], [500, 'stripe_secret_key_missing']]);
    expect(logs[1].event_id).toBe('evt_nokey2');
  });
  it('setup mode and unpaid sessions are refused', async () => {
    const setup = evt({ id: 'evt_fb3' }); setup.data.object.mode = 'setup';
    expect(await (await worker.fetch(await signed(setup), env)).json()).toMatchObject({ ignored: true, reason: 'checkout_mode_setup' });
    const unpaid = evt({ id: 'evt_fb4' }); unpaid.data.object.payment_status = 'unpaid';
    expect(await (await worker.fetch(await signed(unpaid), env)).json()).toMatchObject({ ignored: true, reason: 'payment_status_unpaid' });
    expect(bal()).toBe(1000);
  });
  it('tampered metadata.credits is ignored; server value granted and mismatch logged', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const e = evt({ id: 'evt_tamper' }); e.data.object.metadata = { credits: '50000' };
    const j = await (await worker.fetch(await signed(e), env)).json();
    expect(j.granted).toBe(10000); expect(bal()).toBe(11000);
    expect(warn.mock.calls.map((c) => JSON.parse(c[0])).find((x) => x.msg === 'stripe_metadata_credits_mismatch')).toMatchObject({ event_id: 'evt_tamper', metadata_credits: '50000', granted: 10000 });
  });
  it('idempotent replay: granted and ignored events are processed once', async () => {
    const e = evt({ id: 'evt_rep' }); e.data.object.line_items = li('price_test_p25');
    await worker.fetch(await signed(e), env);
    expect(await (await worker.fetch(await signed(e), env)).json()).toMatchObject({ duplicate: true });
    expect(bal()).toBe(26000);
    const bad = evt({ id: 'evt_rep_bad' }); bad.data.object.line_items = li('price_nope');
    await worker.fetch(await signed(bad), env);
    expect(await (await worker.fetch(await signed(bad), env)).json()).toMatchObject({ duplicate: true });
    expect(unmatched()).toHaveLength(1); expect(bal()).toBe(26000);
  });
  it('seller line is intact and no Stripe secret key is committed', async () => {
    const { readFileSync } = await import('node:fs');
    const src = readFileSync(new URL('../src/index.js', import.meta.url), 'utf8');
    const cfg = readFileSync(new URL('../src/config.js', import.meta.url), 'utf8');
    expect(src).toContain('Seller of hosted credits: Autonomicity Games Inc.');
    expect(readFileSync(new URL('../wrangler.toml', import.meta.url), 'utf8')).not.toMatch(/(sk|rk)_(test|live)_[A-Za-z0-9]{10,}/);
    expect(src).not.toMatch(/(sk|rk)_(test|live)_[A-Za-z0-9]{10,}/);
    expect(cfg).not.toMatch(/(sk|rk)_(test|live)_[A-Za-z0-9]{10,}/);
  });
});

describe('LIVE-GATE-1', () => {
  const unmatched = () => env.DB.q('SELECT * FROM unmatched_payments');

  it('LIVE_MODE is fail-closed: only the exact string true is live', () => {
    expect(isLiveMode({})).toBe(false);
    expect(isLiveMode({ LIVE_MODE: undefined })).toBe(false);
    expect(isLiveMode({ LIVE_MODE: 'false' })).toBe(false);
    expect(isLiveMode({ LIVE_MODE: 'TRUE' })).toBe(false);
    expect(isLiveMode({ LIVE_MODE: 'yes' })).toBe(false);
    expect(isLiveMode({ LIVE_MODE: '1' })).toBe(false);
    expect(isLiveMode({ LIVE_MODE: 'true' })).toBe(true);
    expect(stripeKeyMatchesLiveMode({ LIVE_MODE: 'true', STRIPE_SECRET_KEY: SK_LIVE })).toBe(true);
    expect(stripeKeyMatchesLiveMode({ LIVE_MODE: 'true', STRIPE_SECRET_KEY: RK_LIVE })).toBe(true);
    expect(stripeKeyMatchesLiveMode({ LIVE_MODE: 'true', STRIPE_SECRET_KEY: SK_TEST })).toBe(false);
    expect(stripeKeyMatchesLiveMode({ STRIPE_SECRET_KEY: SK_TEST })).toBe(true);
    expect(stripeKeyMatchesLiveMode({ LIVE_MODE: 'false', STRIPE_SECRET_KEY: SK_LIVE })).toBe(false);
  });

  it('(a) test-mode event against LIVE_MODE=true grants 0 credits, writes unmatched_payments, returns 200', async () => {
    stripeFetch();
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    env.LIVE_MODE = 'true';
    env.STRIPE_SECRET_KEY = SK_LIVE;
    const r = await worker.fetch(await signed(evt({ id: 'evt_test_in_live' })), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'livemode_mismatch' });
    expect(bal()).toBe(1000);
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_test_in_live', reason: 'livemode_mismatch' }]);
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).find((x) => x.msg === 'livemode_mismatch')).toMatchObject({ alert: true, event_livemode: false, live_mode: true });
  });

  it('(d) LIVE_MODE unset behaves as sandbox', async () => {
    delete env.LIVE_MODE;
    stripeFetch();
    const f = vi.mocked(globalThis.fetch);
    const liveEvt = evt({ id: 'evt_unset_live', livemode: true });
    const r = await worker.fetch(await signed(liveEvt), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'livemode_mismatch' });
    expect(bal()).toBe(1000);
    env.STRIPE_SECRET_KEY = SK_LIVE;
    const mm = await worker.fetch(await signed(evt({ id: 'evt_unset_livekey' })), env);
    expect(mm.status).toBe(500);
    expect(await mm.json()).toMatchObject({ error: 'stripe_key_mode_mismatch' });
    expect(f.mock.calls.filter((c) => String(c[0]).includes('api.stripe.com'))).toHaveLength(0);
    env.STRIPE_SECRET_KEY = SK_TEST;
    expect(await (await worker.fetch(await signed(evt({ id: 'evt_unset_ok' })), env)).json()).toMatchObject({ granted: 10000 });
    expect(bal()).toBe(11000);
  });

  it('WEB-AUTH-1: chat does not require Stripe; missing or mismatched key is not 500', async () => {
    const f = okFetch();
    delete env.STRIPE_SECRET_KEY;
    expect((await worker.fetch(req(), env)).status).toBe(200);
    env.LIVE_MODE = 'true';
    env.MERCY_CHECKS_LEVEL = 'launch'; // MERCY-LAUNCH-1: production-like env needs the launch level
    env.STRIPE_SECRET_KEY = SK_TEST;
    expect((await worker.fetch(req(), env)).status).toBe(200);
    expect(f).toHaveBeenCalled();
    expect(bal()).toBeLessThan(1000);
  });

  it('(e) max_tokens above 4096 is clamped; hold is worst case; insufficient balance is 402 with no ledger change; settlement never exceeds hold', async () => {
    expect(MAX_TOKENS_CAP).toBe(4096);
    expect(DEFAULT_MAX_TOKENS).toBe(4096);
    const body = { model: 'grok-4.7', max_tokens: 99999, messages: [{ role: 'user', content: 'hi' }] };
    const worst = holdEstimate(body.model, new TextEncoder().encode(JSON.stringify(body.messages)).length, 1, MAX_TOKENS_CAP);
    expect(worst).toBe(holdEstimate(body.model, new TextEncoder().encode(JSON.stringify(body.messages)).length, 1, 4096));

    env.DB.q("UPDATE accounts SET balance = 1 WHERE id='u1'");
    const f = okFetch();
    const refused = await worker.fetch(req(body), env);
    expect(refused.status).toBe(402);
    expect(await refused.json()).toMatchObject({ error: 'insufficient_credits', required_hold: worst });
    expect(f).not.toHaveBeenCalled();
    expect(bal()).toBe(1);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM ledger')[0].n).toBe(0);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM holds')[0].n).toBe(0);

    env.DB.q("UPDATE accounts SET balance = 1000 WHERE id='u1'");
    let heldDuringCall;
    vi.mocked(globalThis.fetch).mockRestore?.();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
      heldDuringCall = env.DB.q("SELECT amount FROM holds WHERE status='open'")[0]?.amount;
      const sent = JSON.parse(init.body);
      expect(sent.max_tokens).toBe(4096);
      return new Response(JSON.stringify({ choices: [], usage: { prompt_tokens: 100, completion_tokens: 4096 } }));
    });
    const j = await (await worker.fetch(req(body), env)).json();
    expect(heldDuringCall).toBe(worst);
    expect(j.rathor.credits_held).toBe(worst);
    expect(j.rathor.credits_charged).toBe(creditsForUsage(body.model, { prompt_tokens: 100, completion_tokens: 4096 }));
    expect(j.rathor.credits_charged).toBeLessThanOrEqual(j.rathor.credits_held);
    expect(bal()).toBe(1000 - j.rathor.credits_charged);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM holds WHERE status='open'")[0].n).toBe(0);
  });

  it('README worst-case holds match credits.js at MAX_TOKENS_CAP and must be re-derived when rates flip', async () => {
    const { readFileSync } = await import('node:fs');
    const md = readFileSync(new URL('../README.md', import.meta.url), 'utf8');
    expect(MAX_TOKENS_CAP).toBe(4096);
    const h47 = holdEstimate('grok-4.7', 0, 1, MAX_TOKENS_CAP);
    const h43 = holdEstimate('grok-4.3', 0, 1, MAX_TOKENS_CAP);
    expect(h47).toBe(179);
    expect(h43).toBe(75);
    expect(md).toMatch(/`grok-4\.7`\s*\|\s*\*\*179 credits\*\*/);
    expect(md).toMatch(/`grok-4\.3`\s*\|\s*\*\*75 credits\*\*/);
    expect(md).toContain('Re-derive the worst-case hold table');
    expect(md).toContain('[env.production]');
    expect(md).toContain('LIVE_MODE');
  });

  it('wrangler.toml keeps sandbox LIVE_MODE false and production LIVE_MODE false with its own D1', async () => {
    const { readFileSync } = await import('node:fs');
    const toml = readFileSync(new URL('../wrangler.toml', import.meta.url), 'utf8');
    expect(toml).toMatch(/\[vars\][\s\S]*LIVE_MODE = "false"/);
    expect(toml).toMatch(/\[vars\][\s\S]*STRIPE_RETRY_EXHAUST_SECONDS = "21600"/);
    expect(toml).toMatch(/\[env\.production\.vars\][\s\S]*LIVE_MODE = "false"/);
    expect(toml).toMatch(/\[env\.production\.vars\][\s\S]*STRIPE_RETRY_EXHAUST_SECONDS = "262800"/);
    expect(toml).toContain('[[env.production.d1_databases]]');
    expect(toml).toContain('database_name = "rathor-api-production"');
    expect(toml).toContain('database_id = "PLACEHOLDER_NOT_CREATED"');
    expect(toml).toContain('wrangler secret put --env production');
    expect(toml).not.toMatch(/(sk|rk)_(test|live)_[A-Za-z0-9]{10,}/);
  });
});
