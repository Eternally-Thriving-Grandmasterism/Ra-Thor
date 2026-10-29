import { describe, it, expect, vi, beforeEach } from 'vitest';
import worker from '../src/index.js';
import { hmacHex } from '../src/stripe.js';
import { sha256Hex } from '../src/auth.js';
import { fakeD1 } from './fakeD1.js';

// Placeholders built at runtime so no key literal appears in the test file.
const sk = (mode, tail = 'PLACEHOLDER') => ['sk', mode, tail].join('_');
const WHSEC = ['whsec', 'test'].join('_');
const SK_TEST = sk('test', 'FAKE_not_a_real_key');

const O = 'https://rathor.ai';
const PRICES = { STRIPE_PRICE_PACK_10: 'price_test_p10', STRIPE_PRICE_PACK_25: 'price_test_p25', STRIPE_PRICE_PACK_50: 'price_test_p50', STRIPE_PRICE_PLAN_MONTHLY: 'price_test_plan', STRIPE_PRICE_TRIAL: 'price_test_trial' };
const KEY = ['rth', 'test', '0123456789abcdefghijklmnop'].join('_');
const STRIPE_API_OBJECTS = new Map();
let env;

beforeEach(async () => {
  vi.restoreAllMocks();
  STRIPE_API_OBJECTS.clear();
  env = { ALLOWED_ORIGIN: O, DB: fakeD1(), XAI_API_KEY: 'test', STRIPE_WEBHOOK_SECRET: WHSEC, RESEND_API_KEY: 'test', RATES_VERIFIED: 'true', STRIPE_SECRET_KEY: SK_TEST, ...PRICES };
  env.DB.q("INSERT INTO accounts (id, balance) VALUES ('u1', 1000)");
  env.DB.q("INSERT INTO ledger (account_id, delta, reason) VALUES ('u1', 1000, 'seed')");
  env.DB.q("UPDATE accounts SET subscription_id='sub_1', subscription_status='active', stripe_customer_id='cus_1' WHERE id='u1'");
  env.DB.q('INSERT INTO api_keys (key_hash, account_id) VALUES (?1, ?2)', await sha256Hex(KEY), 'u1');
});

const bal = (id = 'u1') => env.DB.q('SELECT balance FROM accounts WHERE id = ?1', id)[0]?.balance;
const frozen = (id = 'u1') => env.DB.q('SELECT frozen, freeze_reason FROM accounts WHERE id = ?1', id)[0];
const unmatched = () => env.DB.q('SELECT * FROM unmatched_payments');
const grants = () => env.DB.q('SELECT * FROM credit_grants');

async function signed(evt, secret = WHSEC) {
  const body = JSON.stringify(evt);
  const t = Math.floor(Date.now() / 1000);
  return new Request('https://api.rathor.ai/v1/stripe/webhook', { method: 'POST', body, headers: { 'Stripe-Signature': `t=${t},v1=${await hmacHex(secret, `${t}.${body}`)}` } });
}

const stripeFetch = (onMatch) => vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
  const u = String(url);
  if (onMatch) {
    const override = await onMatch(u, init);
    if (override) return override;
  }
  const inpay = u.match(/^https:\/\/api\.stripe\.com\/v1\/invoice_payments\?(.*)$/);
  if (inpay) {
    const invoice = new URLSearchParams(inpay[1]).get('invoice');
    const obj = STRIPE_API_OBJECTS.get(`inpay:${invoice}`);
    return obj ? new Response(JSON.stringify(obj)) : new Response(JSON.stringify({ object: 'list', data: [], has_more: false }));
  }
  const pi = u.match(/^https:\/\/api\.stripe\.com\/v1\/payment_intents\/([^/?]+)/);
  if (pi) {
    const obj = STRIPE_API_OBJECTS.get(`pi:${decodeURIComponent(pi[1])}`);
    return obj ? new Response(JSON.stringify(obj)) : new Response('{"error":{"type":"invalid_request_error"}}', { status: 404 });
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

function putPaymentIntent(id, fields) {
  const obj = { id, object: 'payment_intent', ...fields };
  STRIPE_API_OBJECTS.set(`pi:${id}`, obj);
  return obj;
}

function putInvoicePayments(invoiceId, data) {
  const obj = { object: 'list', has_more: false, data };
  STRIPE_API_OBJECTS.set(`inpay:${invoiceId}`, obj);
  return obj;
}

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

function paidInvoicePayment(pi) {
  return { id: 'inpay_1', object: 'invoice_payment', status: 'paid', payment: { type: 'payment_intent', payment_intent: pi } };
}

// Basil-shaped invoice.paid: api_version 2025-05-28.basil, no top-level
// payment_intent / charge / subscription_details. subId comes from parent.
function basilInvoice(id, extras = {}) {
  const invoiceId = extras.invoiceId ?? `in_${id}`;
  const payments = extras.payments;
  const object = {
    id: invoiceId,
    object: 'invoice',
    customer: 'cus_1',
    customer_email: 't@example.com',
    billing_reason: 'subscription_cycle',
    amount_paid: extras.amount_paid ?? 1000,
    currency: 'usd',
    parent: {
      type: 'subscription_details',
      subscription_details: {
        subscription: 'sub_1',
        metadata: extras.metadata ?? { rathor_user: 'u1' },
      },
    },
    lines: { data: [{ quantity: 1, currency: 'usd', price: { id: 'price_test_plan', unit_amount: 1000, currency: 'usd' } }] },
  };
  if (payments !== undefined) object.payments = payments;
  else if (extras.paymentIntent) {
    object.payments = { object: 'list', has_more: false, data: [paidInvoicePayment(extras.paymentIntent)] };
  }
  return {
    id,
    object: 'event',
    api_version: '2025-05-28.basil',
    type: 'invoice.paid',
    livemode: false,
    data: { object },
  };
}

function oldShapeInvoice(id, { pi = `pi_${id}`, charge = `ch_${id}` } = {}) {
  return {
    id,
    type: 'invoice.paid',
    livemode: false,
    data: {
      object: {
        subscription: 'sub_1',
        customer: 'cus_1',
        customer_email: 't@example.com',
        billing_reason: 'subscription_cycle',
        amount_paid: 1000,
        currency: 'usd',
        payment_intent: pi,
        charge,
        subscription_details: { metadata: { rathor_user: 'u1' } },
        lines: { data: [{ quantity: 1, currency: 'usd', price: { id: 'price_test_plan', unit_amount: 1000, currency: 'usd' } }] },
      },
    },
  };
}

const chatReq = () =>
  new Request('https://api.rathor.ai/v1/chat', { method: 'POST', body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }), headers: { Origin: O, Authorization: `Bearer ${KEY}` } });

async function grantBasil({ id = 'evt_basil_g1', pi = 'pi_basil_1', charge = 'ch_basil_1' } = {}) {
  putPaymentIntent(pi, { latest_charge: charge });
  stripeFetch();
  const j = await (await worker.fetch(await signed(basilInvoice(id, { paymentIntent: pi })), env)).json();
  expect(j.granted).toBe(10000);
  return { pi, charge };
}

describe('SPROOF-FIX-RENEWAL-IDS basil invoice.paid', () => {
  it('renewal grant stores PaymentIntent and charge from payments.data + latest_charge', async () => {
    const f = stripeFetch();
    putPaymentIntent('pi_basil_1', { latest_charge: 'ch_basil_1' });
    const evt = basilInvoice('evt_basil_g1', { paymentIntent: 'pi_basil_1' });
    expect(evt.data.object.payment_intent).toBeUndefined();
    expect(evt.data.object.charge).toBeUndefined();
    expect(evt.data.object.subscription_details).toBeUndefined();
    const r = await worker.fetch(await signed(evt), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ granted: 10000, kind: 'plan' });
    expect(bal()).toBe(11000);
    expect(grants()).toMatchObject([{
      account_id: 'u1',
      stripe_event_id: 'evt_basil_g1',
      payment_intent_id: 'pi_basil_1',
      charge_id: 'ch_basil_1',
      granted_credits: 10000,
      amount_paid: 1000,
    }]);
    const urls = f.mock.calls.map((c) => String(c[0]));
    expect(urls.some((u) => u.includes('/v1/invoice_payments'))).toBe(false);
    expect(urls).toContain('https://api.stripe.com/v1/payment_intents/pi_basil_1');
    expect(grants()[0].payment_intent_id).not.toBeNull();
    expect(grants()[0].charge_id).not.toBeNull();
  });

  it('full refund on a basil renewal claws the granted credits', async () => {
    await grantBasil();
    expect(bal()).toBe(11000);
    const charge = putCharge('ch_basil_1', { amount: 1000, amount_refunded: 1000, payment_intent: 'pi_basil_1' });
    const j = await (await worker.fetch(await signed({ id: 'evt_basil_rf_full', type: 'charge.refunded', livemode: false, data: { object: charge } }), env)).json();
    expect(j.clawed).toBe(10000);
    expect(j.shortfall).toBe(0);
    expect(bal()).toBe(1000);
    expect(grants()[0].clawed_credits).toBe(10000);
    expect(env.DB.q("SELECT delta, reason FROM ledger WHERE stripe_event_id='evt_basil_rf_full'")[0]).toMatchObject({ delta: -10000, reason: 'refund_claw' });
  });

  it('partial refund on a basil renewal claws a rounded-up share', async () => {
    putPaymentIntent('pi_basil_part', { latest_charge: 'ch_basil_part' });
    stripeFetch();
    const evt = basilInvoice('evt_basil_part', { paymentIntent: 'pi_basil_part', amount_paid: 3000 });
    expect(await (await worker.fetch(await signed(evt), env)).json()).toMatchObject({ granted: 10000 });
    const charge = putCharge('ch_basil_part', { amount: 3000, amount_refunded: 1000, payment_intent: 'pi_basil_part' });
    const j = await (await worker.fetch(await signed({ id: 'evt_basil_rf_part', type: 'charge.refunded', livemode: false, data: { object: charge } }), env)).json();
    expect(j.target).toBe(3334);
    expect(j.clawed).toBe(3334);
    expect(bal()).toBe(11000 - 3334);
    expect(grants()[0].clawed_credits).toBe(3334);
  });

  it('dispute on a basil renewal freezes the account', async () => {
    await grantBasil();
    const d = putDispute('dp_basil_1', { status: 'needs_response', charge: 'ch_basil_1', payment_intent: 'pi_basil_1', customer: 'cus_1' });
    const j = await (await worker.fetch(await signed({ id: 'evt_basil_dp', type: 'charge.dispute.created', livemode: false, data: { object: d } }), env)).json();
    expect(j.frozen).toBe(true);
    expect(frozen()).toMatchObject({ frozen: 1, freeze_reason: 'dispute' });
    const chat = await worker.fetch(chatReq(), env);
    expect(chat.status).toBe(403);
    expect(await chat.json()).toMatchObject({ error: 'account_frozen' });
  });

  it('missing payment ids refuse() and never store a NULL grant', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const f = stripeFetch();
    putInvoicePayments('in_evt_basil_noid', []);
    const evt = basilInvoice('evt_basil_noid', { payments: { object: 'list', has_more: false, data: [] } });
    const r = await worker.fetch(await signed(evt), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'renewal_payment_ids_missing' });
    expect(bal()).toBe(1000);
    expect(grants()).toHaveLength(0);
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_basil_noid', reason: 'renewal_payment_ids_missing' }]);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE stripe_event_id='evt_basil_noid'")[0].n).toBe(0);
    expect(f.mock.calls.map((c) => String(c[0]))).toContain('https://api.stripe.com/v1/invoice_payments?invoice=in_evt_basil_noid');
    expect(warn.mock.calls.map((c) => JSON.parse(c[0])).find((x) => x.reason === 'renewal_payment_ids_missing')).toMatchObject({ alert: true });
  });

  it('invoice_payments fallback supplies the PaymentIntent when payments is absent', async () => {
    const f = stripeFetch();
    putInvoicePayments('in_evt_basil_fb', [paidInvoicePayment('pi_basil_fb')]);
    putPaymentIntent('pi_basil_fb', { latest_charge: 'ch_basil_fb' });
    const evt = basilInvoice('evt_basil_fb', { payments: undefined });
    delete evt.data.object.payments;
    const r = await worker.fetch(await signed(evt), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ granted: 10000 });
    expect(grants()).toMatchObject([{ payment_intent_id: 'pi_basil_fb', charge_id: 'ch_basil_fb' }]);
    const urls = f.mock.calls.map((c) => String(c[0]));
    expect(urls).toContain('https://api.stripe.com/v1/invoice_payments?invoice=in_evt_basil_fb');
    expect(urls).toContain('https://api.stripe.com/v1/payment_intents/pi_basil_fb');
  });

  it('exhausted PaymentIntent 404 writes one stripe_fetch_exhausted row and no grant', async () => {
    const NOW_SEC = 1_700_000_000;
    vi.spyOn(Date, 'now').mockReturnValue(NOW_SEC * 1000);
    env.STRIPE_RETRY_EXHAUST_SECONDS = '21600';
    stripeFetch();
    const evt = basilInvoice('evt_basil_exh', { paymentIntent: 'pi_basil_exh' });
    evt.created = NOW_SEC - 21600;
    const r = await worker.fetch(await signed(evt), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ ignored: true, reason: 'stripe_fetch_exhausted' });
    expect(unmatched()).toMatchObject([{ stripe_event_id: 'evt_basil_exh', reason: 'stripe_fetch_exhausted', detail: '404' }]);
    expect(grants()).toHaveLength(0);
    expect(bal()).toBe(1000);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_basil_exh'")[0].n).toBe(0);
  });

  it('transient PaymentIntent fetch returns 5xx and writes nothing', async () => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    stripeFetch(async (u) => {
      if (u.includes('/v1/payment_intents/')) return new Response('{"error":{}}', { status: 500 });
      return undefined;
    });
    const evt = basilInvoice('evt_basil_pi502', { paymentIntent: 'pi_basil_502' });
    const r = await worker.fetch(await signed(evt), env);
    expect(r.status).toBe(502);
    expect(await r.json()).toMatchObject({ error: 'stripe_fetch_transient', retry: true, status: 500 });
    expect(bal()).toBe(1000);
    expect(grants()).toHaveLength(0);
    expect(unmatched()).toHaveLength(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM stripe_events WHERE event_id='evt_basil_pi502'")[0].n).toBe(0);
    expect(env.DB.q("SELECT COUNT(*) AS n FROM ledger WHERE stripe_event_id='evt_basil_pi502'")[0].n).toBe(0);
    expect(err.mock.calls.map((c) => JSON.parse(c[0])).find((x) => x.msg === 'stripe_fetch_transient')).toMatchObject({ alert: true, status: 500 });
  });

  it('pre-basil invoices still grant from top-level payment_intent and charge', async () => {
    const f = stripeFetch();
    const r = await worker.fetch(await signed(oldShapeInvoice('evt_old_1')), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toMatchObject({ granted: 10000 });
    expect(grants()).toMatchObject([{ payment_intent_id: 'pi_evt_old_1', charge_id: 'ch_evt_old_1' }]);
    const urls = f.mock.calls.map((c) => String(c[0]));
    expect(urls.some((u) => u.includes('/v1/invoice_payments'))).toBe(false);
    expect(urls.some((u) => u.includes('/v1/payment_intents/'))).toBe(false);
  });
});
