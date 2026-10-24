// Ra-Thor API Worker for api.rathor.ai — SANDBOX DRAFT, not deployed.
// Ported from app/api/grok/route.js. Hosted credits sold by Autonomicity Games Inc.
// All secrets come from Worker secrets; none live in this repo.
import { creditsForUsage, getBalance, decrement, grant, PLACEHOLDER_PRICING } from './credits.js';
import { verifyStripeSignature } from './stripe.js';

const XAI_URL = 'https://api.x.ai/v1/chat/completions';
const RESEND_URL = 'https://api.resend.com/emails';
export const RECEIPT_FROM = 'Ra-Thor <info@rathor.ai>';

function cors(env, req) {
  const allowed = env.ALLOWED_ORIGIN || 'https://rathor.ai';
  const h = { 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type, Authorization', Vary: 'Origin' };
  if (req.headers.get('Origin') === allowed) h['Access-Control-Allow-Origin'] = allowed;
  return h;
}
const json = (obj, status, headers = {}) => new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json', ...headers } });

// PLACEHOLDER auth: bearer token mapped to a user id in KV (`token:<t>` -> userId).
async function userFrom(req, env) {
  const m = /^Bearer (.+)$/.exec(req.headers.get('Authorization') || '');
  if (!m) return null;
  return env.CREDITS.get(`token:${m[1]}`);
}

async function chat(req, env, h) {
  if (req.headers.get('Origin') !== (env.ALLOWED_ORIGIN || 'https://rathor.ai')) return json({ error: 'origin_not_allowed' }, 403, h);
  const user = await userFrom(req, env);
  if (!user) return json({ error: 'unauthorized' }, 401, h);
  if ((await getBalance(env.CREDITS, user)) <= 0) return json({ error: 'insufficient_credits' }, 402, h);
  let body;
  try { body = await req.json(); } catch { return json({ error: 'bad_json' }, 400, h); }
  const r = await fetch(XAI_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.XAI_API_KEY}` },
    body: JSON.stringify({ ...body, stream: false }),
  });
  if (!r.ok) return json({ error: 'upstream_error', status: r.status }, 502, h);
  const data = await r.json();
  const cost = creditsForUsage(data.usage);
  const remaining = await decrement(env.CREDITS, user, cost);
  return json({ ...data, rathor: { credits_charged: cost, credits_remaining: remaining, pricing_status: PLACEHOLDER_PRICING.status } }, 200, h);
}

async function sendReceipt(env, to, credits) {
  if (!env.RESEND_API_KEY || !to) return false;
  const r = await fetch(RESEND_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.RESEND_API_KEY}` },
    body: JSON.stringify({ from: RECEIPT_FROM, to: [to], subject: 'Ra-Thor credits receipt (test mode)', text: `Test-mode receipt: ${credits} credits were added to your Ra-Thor account. Sold by Autonomicity Games Inc.` }),
  });
  return r.ok;
}

async function webhook(req, env) {
  const payload = await req.text();
  const ok = await verifyStripeSignature(payload, req.headers.get('Stripe-Signature'), env.STRIPE_WEBHOOK_SECRET);
  if (!ok) return json({ error: 'bad_signature' }, 400);
  const event = JSON.parse(payload);
  if (event.livemode === true) return json({ error: 'livemode_refused_test_mode_only' }, 403);
  if (event.type !== 'checkout.session.completed') return json({ received: true, ignored: event.type }, 200);
  const s = event.data?.object || {};
  if (s.livemode === true) return json({ error: 'livemode_refused_test_mode_only' }, 403);
  const user = s.client_reference_id || s.metadata?.rathor_user;
  if (!user) return json({ error: 'no_user' }, 400);
  if (await env.CREDITS.get(`evt:${event.id}`)) return json({ received: true, duplicate: true }, 200);
  const credits = parseInt(s.metadata?.credits, 10) || PLACEHOLDER_PRICING.monthlyPlanCredits;
  await env.CREDITS.put(`evt:${event.id}`, '1');
  const balance = await grant(env.CREDITS, user, credits);
  const emailed = await sendReceipt(env, s.customer_details?.email, credits);
  return json({ received: true, granted: credits, balance, emailed }, 200);
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const h = cors(env, req);
    if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: h });
    if (req.method === 'GET' && url.pathname === '/health') return json({ ok: true, service: 'rathor-api', mode: 'sandbox' }, 200, h);
    if (req.method === 'POST' && url.pathname === '/v1/chat') return chat(req, env, h);
    if (req.method === 'POST' && url.pathname === '/v1/stripe/webhook') return webhook(req, env);
    return json({ error: 'not_found' }, 404, h);
  },
};
