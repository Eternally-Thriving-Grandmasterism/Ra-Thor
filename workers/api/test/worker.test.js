import { describe, it, expect, vi, beforeEach } from 'vitest';
import worker from '../src/index.js';
import { hmacHex, verifyStripeSignature } from '../src/stripe.js';
import { creditsForUsage } from '../src/credits.js';

const kv = () => { const m = new Map(); return { get: async (k) => (m.has(k) ? m.get(k) : null), put: async (k, v) => void m.set(k, v), m }; };
const O = 'https://rathor.ai';
let env;
beforeEach(() => { env = { ALLOWED_ORIGIN: O, CREDITS: kv(), XAI_API_KEY: 'test', STRIPE_WEBHOOK_SECRET: 'whsec_test', RESEND_API_KEY: 'test' }; vi.restoreAllMocks(); });

async function signed(evt, secret = 'whsec_test') {
  const body = JSON.stringify(evt); const t = Math.floor(Date.now() / 1000);
  return new Request('https://api.rathor.ai/v1/stripe/webhook', { method: 'POST', body, headers: { 'Stripe-Signature': `t=${t},v1=${await hmacHex(secret, `${t}.${body}`)}` } });
}
const evt = (o = {}) => ({ id: 'evt_1', type: 'checkout.session.completed', livemode: false, data: { object: { client_reference_id: 'u1', metadata: {}, customer_details: { email: 'test@example.com' } } }, ...o });

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

describe('webhook', () => {
  it('grants credits and emails from info@rathor.ai', async () => {
    const f = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{}', { status: 200 }));
    const r = await worker.fetch(await signed(evt()), env);
    const j = await r.json();
    expect(r.status).toBe(200); expect(j.balance).toBe(550);
    expect(JSON.parse(f.mock.calls[0][1].body).from).toBe('Ra-Thor <info@rathor.ai>');
    const again = await (await worker.fetch(await signed(evt()), env)).json();
    expect(again.duplicate).toBe(true);
  });
  it('refuses livemode', async () => { expect((await worker.fetch(await signed(evt({ livemode: true })), env)).status).toBe(403); });
  it('rejects wrong secret', async () => { expect((await worker.fetch(await signed(evt(), 'nope'), env)).status).toBe(400); });
});

describe('chat', () => {
  const req = (origin = O) => new Request('https://api.rathor.ai/v1/chat', { method: 'POST', body: '{"messages":[]}', headers: { Origin: origin, Authorization: 'Bearer tok' } });
  it('blocks other origins', async () => { expect((await worker.fetch(req('https://evil.example'), env)).status).toBe(403); });
  it('401 without user', async () => { expect((await worker.fetch(req(), env)).status).toBe(401); });
  it('charges credits', async () => {
    await env.CREDITS.put('token:tok', 'u1');
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ usage: { prompt_tokens: 1000, completion_tokens: 1000 } })));
    const r = await worker.fetch(req(), env); const j = await r.json();
    expect(r.status).toBe(200); expect(j.rathor.credits_charged).toBe(3); expect(j.rathor.credits_remaining).toBe(97);
    expect(r.headers.get('Access-Control-Allow-Origin')).toBe(O);
  });
  it('402 when empty', async () => {
    await env.CREDITS.put('token:tok', 'u1'); await env.CREDITS.put('credits:u1', '0');
    expect((await worker.fetch(req(), env)).status).toBe(402);
  });
  it('min 1 credit', () => { expect(creditsForUsage({})).toBe(1); });
});
