import { describe, it, expect, vi, beforeEach } from 'vitest';
import worker from '../src/index.js';
import { sha256Hex } from '../src/auth.js';
import { SESSION_COOKIE_NAME, CSRF_HEADER, CHAT_MESSAGES_MAX_BYTES, AUTH_REQUEST_PER_EMAIL_PER_HOUR, AUTH_REQUEST_PER_IP_PER_HOUR, AUTH_TOKEN_TTL_SEC, stripeKeyPrefixMode, stripeKeyMatchesLiveMode, CUSTOMER_KEY_PREFIX_TEST } from '../src/config.js';
import { SIGNUP_GRANT_CREDITS } from '../src/db.js';
import { randomToken, AUTH_TOKEN_RE, AUTH_TOKEN_LENGTH, AUTH_TOKEN_BYTES, VERIFY_FAIL_MESSAGE, VERIFY_PAGE_REFERRER_POLICY, allowedOrigin, escapeHtmlAttr } from '../src/session.js';
import { fakeD1 } from './fakeD1.js';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const sk = (mode, tail = 'PLACEHOLDER') => ['sk', mode, tail].join('_');
const SK_TEST = sk('test');
const SK_LIVE = sk('live');
const O = 'https://rathor.ai';
const PRICES = { STRIPE_PRICE_PACK_10: 'price_test_p10', STRIPE_PRICE_PACK_25: 'price_test_p25', STRIPE_PRICE_PACK_50: 'price_test_p50', STRIPE_PRICE_PLAN_MONTHLY: 'price_test_plan', STRIPE_PRICE_TRIAL: 'price_test_trial' };
const KEY = ['rth', 'test', '0123456789abcdefghijklmnop'].join('_');
const CHECKOUT_URL = 'https://checkout.stripe.com/c/pay/cs_test_placeholder';

let env;
beforeEach(async () => {
  vi.restoreAllMocks();
  env = { ALLOWED_ORIGIN: O, DB: fakeD1(), XAI_API_KEY: 'test', STRIPE_WEBHOOK_SECRET: 'whsec_test', RESEND_API_KEY: 'test', RATES_VERIFIED: 'true', STRIPE_SECRET_KEY: SK_TEST, ...PRICES };
  env.DB.q("INSERT INTO accounts (id, balance) VALUES ('u1', 1000)");
  env.DB.q('INSERT INTO api_keys (key_hash, account_id) VALUES (?1, ?2)', await sha256Hex(KEY), 'u1');
});

const bal = (id = 'u1') => env.DB.q('SELECT balance FROM accounts WHERE id = ?1', id)[0]?.balance;

async function putSession(accountId = 'u1') {
  const raw = ['sess', crypto.randomUUID().replace(/-/g, ''), 'token'].join('');
  const exp = Math.floor(Date.now() / 1000) + 86400;
  env.DB.q('INSERT INTO sessions (session_hash, account_id, expires_at) VALUES (?1, ?2, ?3)', await sha256Hex(raw), accountId, exp);
  return raw;
}

function headers({ origin = O, cookie, csrf = '1', extra = {} } = {}) {
  const h = { Origin: origin, ...extra };
  if (cookie) h.Cookie = `${SESSION_COOKIE_NAME}=${cookie}`;
  if (csrf) h[CSRF_HEADER] = csrf;
  return h;
}

const post = (path, body, opts = {}) =>
  new Request(`https://api.rathor.ai${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers(opts) }, body: body === undefined ? undefined : JSON.stringify(body) });

function formPost(token, { origin = 'https://api.rathor.ai', csrf = '1' } = {}) {
  const h = { 'Content-Type': 'application/x-www-form-urlencoded' };
  if (origin !== null) h.Origin = origin;
  const body = new URLSearchParams();
  if (token !== undefined) body.set('token', token);
  if (csrf !== null) body.set('csrf', csrf);
  return new Request('https://api.rathor.ai/v1/auth/verify', { method: 'POST', headers: h, body: body.toString() });
}

function expectVerifyPageHeaders(r) {
  const dest = allowedOrigin(env);
  const csp = r.headers.get('Content-Security-Policy');
  expect(csp).toContain(`form-action 'self' ${dest}`);
  expect(csp).toContain("script-src 'none'");
  expect(csp).toContain("connect-src 'none'");
  expect(r.headers.get('X-Frame-Options')).toBe('DENY');
  expect(r.headers.get('Referrer-Policy')).not.toBe('no-referrer');
  expect(r.headers.get('Referrer-Policy')).toBe(VERIFY_PAGE_REFERRER_POLICY);
  expect(VERIFY_PAGE_REFERRER_POLICY).toBe('same-origin');
  expect(r.headers.get('Cache-Control')).toBe('no-store');
}

function extractToken(fetchMock) {
  const call = fetchMock.mock.calls.find((c) => String(c[0]).includes('resend.com'));
  expect(call).toBeTruthy();
  const sent = JSON.parse(call[1].body);
  const m = /token=([A-Za-z0-9_\-]+)/.exec(sent.text);
  expect(m).toBeTruthy();
  return { token: m[1], sent };
}

function stripeCheckoutFetch(inspect) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
    if (String(url) === 'https://api.stripe.com/v1/checkout/sessions') {
      if (inspect) inspect(new URLSearchParams(init.body), init);
      return new Response(JSON.stringify({ id: 'cs_test_x', url: CHECKOUT_URL, livemode: false }));
    }
    if (String(url).includes('resend.com')) return new Response('{}');
    return new Response('{}');
  });
}

const okChatFetch = (usage = { prompt_tokens: 1000, completion_tokens: 500 }) =>
  vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(JSON.stringify({ choices: [{ message: { content: 'hello' } }], usage })));

describe('WEB-AUTH-1 magic link', () => {
  it('sends a link from Ra-Thor <info@rathor.ai>, stores only the SHA-256 hash, same body whether the email is new or known', async () => {
    const f = stripeCheckoutFetch();
    const r1 = await worker.fetch(post('/v1/auth/request', { email: 'New.User@Example.com' }), env);
    expect(r1.status).toBe(200);
    expect(await r1.json()).toEqual({ ok: true });
    const { token, sent } = extractToken(f);
    expect(sent.from).toBe('Ra-Thor <info@rathor.ai>');
    expect(sent.to).toEqual(['new.user@example.com']);
    expect(token.length).toBeGreaterThanOrEqual(43);
    const rows = env.DB.q('SELECT token_hash, email FROM auth_tokens');
    expect(rows).toHaveLength(1);
    expect(rows[0].token_hash).toBe(await sha256Hex(token));
    expect(rows[0].token_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(JSON.stringify(rows)).not.toContain(token);
    expect(rows[0].email).toBe('new.user@example.com');

    env.DB.q("UPDATE accounts SET email = 'known@example.com' WHERE id = 'u1'");
    const r2 = await worker.fetch(post('/v1/auth/request', { email: 'known@example.com' }), env);
    expect(r2.status).toBe(200);
    expect(await r2.json()).toEqual({ ok: true });
  });

  it('generated tokens match AUTH_TOKEN_RE derived from the 32-byte generator', () => {
    expect(AUTH_TOKEN_BYTES).toBe(32);
    expect(AUTH_TOKEN_LENGTH).toBe(43);
    expect(AUTH_TOKEN_RE.source).toBe(`^[A-Za-z0-9_-]{${AUTH_TOKEN_LENGTH}}$`);
    const token = randomToken();
    expect(token).toMatch(AUTH_TOKEN_RE);
    expect(token.length).toBe(AUTH_TOKEN_LENGTH);
  });

  it('GET verify is a confirm page only: consumed_at stays NULL and no cookie is set', async () => {
    const f = stripeCheckoutFetch();
    await worker.fetch(post('/v1/auth/request', { email: 'fresh@example.com' }), env);
    const { token } = extractToken(f);
    const get = await worker.fetch(new Request(`https://api.rathor.ai/v1/auth/verify?token=${token}`), env);
    expect(get.status).toBe(200);
    expect(get.headers.get('Content-Type')).toMatch(/text\/html/);
    expectVerifyPageHeaders(get);
    const html = await get.text();
    expect(html).toContain('Sign in');
    expect(html).toContain(`value="${token}"`);
    expect(html).toMatch(/method="POST"/i);
    expect(html).toContain('action="/v1/auth/verify"');
    expect(html).not.toMatch(/<script/i);
    expect(get.headers.get('Set-Cookie')).toBeNull();
    expect(env.DB.q('SELECT consumed_at FROM auth_tokens')[0].consumed_at).toBeNull();
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(0);
  });

  it('POST verify from a bad or missing Origin is 403 and does not consume', async () => {
    const f = stripeCheckoutFetch();
    await worker.fetch(post('/v1/auth/request', { email: 'csrf@example.com' }), env);
    const { token } = extractToken(f);
    const hash = await sha256Hex(token);
    const bad = await worker.fetch(post('/v1/auth/verify', { token }, { origin: 'https://evil.example' }), env);
    expect(bad.status).toBe(403);
    expect(await bad.json()).toEqual({ error: 'origin_not_allowed' });
    const missing = await worker.fetch(new Request('https://api.rathor.ai/v1/auth/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', [CSRF_HEADER]: '1' },
      body: JSON.stringify({ token }),
    }), env);
    expect(missing.status).toBe(403);
    expect(env.DB.q('SELECT consumed_at FROM auth_tokens WHERE token_hash = ?1', hash)[0].consumed_at).toBeNull();
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(0);
  });

  it('valid POST verify consumes once, sets a host-only session cookie, new accounts start at 0 credits', async () => {
    const f = stripeCheckoutFetch();
    await worker.fetch(post('/v1/auth/request', { email: 'fresh@example.com' }), env);
    const { token } = extractToken(f);
    const apiOrigin = await worker.fetch(post('/v1/auth/verify', { token }, { origin: 'https://api.rathor.ai' }), env);
    expect(apiOrigin.status).toBe(200);
    expect(await apiOrigin.json()).toEqual({ ok: true });
    const cookie = apiOrigin.headers.get('Set-Cookie');
    expect(cookie).toMatch(new RegExp(`^${SESSION_COOKIE_NAME}=`));
    expect(cookie).toMatch(/HttpOnly/);
    expect(cookie).toMatch(/Secure/);
    expect(cookie).toMatch(/SameSite=Lax/);
    expect(cookie).toMatch(/Path=\//);
    expect(cookie).not.toMatch(/Domain=/i);
    expect(cookie).toMatch(/Max-Age=2592000/);
    const raw = cookie.split(';')[0].split('=')[1];
    expect(env.DB.q('SELECT session_hash FROM sessions')[0].session_hash).toBe(await sha256Hex(raw));
    expect(JSON.stringify(env.DB.q('SELECT * FROM sessions'))).not.toContain(raw);
    expect(env.DB.q('SELECT consumed_at FROM auth_tokens')[0].consumed_at).toBeTruthy();
    const acct = env.DB.q("SELECT id, balance, email FROM accounts WHERE email = 'fresh@example.com'")[0];
    expect(acct.balance).toBe(SIGNUP_GRANT_CREDITS);
    expect(SIGNUP_GRANT_CREDITS).toBe(0);

    const again = await worker.fetch(post('/v1/auth/verify', { token }), env);
    expect(again.status).toBe(400);
    expect(await again.json()).toEqual({ error: 'invalid_token', message: VERIFY_FAIL_MESSAGE });
  });

  it('expired tokens are refused and not consumed into a session', async () => {
    const token = randomToken();
    const past = Math.floor(Date.now() / 1000) - 10;
    env.DB.q('INSERT INTO auth_tokens (token_hash, email, expires_at) VALUES (?1, ?2, ?3)', await sha256Hex(token), 'late@example.com', past);
    const r = await worker.fetch(post('/v1/auth/verify', { token }), env);
    expect(r.status).toBe(400);
    expect(await r.json()).toEqual({ error: 'invalid_token', message: VERIFY_FAIL_MESSAGE });
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(0);
    expect(AUTH_TOKEN_TTL_SEC).toBe(15 * 60);
  });

  it('rate-limits per email (5/hour) and per IP (20/hour) with 429', async () => {
    stripeCheckoutFetch();
    for (let i = 0; i < AUTH_REQUEST_PER_EMAIL_PER_HOUR; i++) {
      expect((await worker.fetch(post('/v1/auth/request', { email: 'same@example.com' }), env)).status).toBe(200);
    }
    const limited = await worker.fetch(post('/v1/auth/request', { email: 'same@example.com' }), env);
    expect(limited.status).toBe(429);
    expect(await limited.json()).toEqual({ error: 'rate_limited' });

    const windowStart = Math.floor(Date.now() / 1000 / 3600) * 3600;
    env.DB.q('INSERT INTO rate_counters (bucket, window_start, count) VALUES (?1, ?2, ?3)', 'ip:203.0.113.9', windowStart, AUTH_REQUEST_PER_IP_PER_HOUR);
    const ipReq = post('/v1/auth/request', { email: 'other@example.com' }, { extra: { 'CF-Connecting-IP': '203.0.113.9' } });
    expect((await worker.fetch(ipReq, env)).status).toBe(429);
  });

  it('ignores X-Forwarded-For and uses CF-Connecting-IP only (or the unknown bucket)', async () => {
    stripeCheckoutFetch();
    const windowStart = Math.floor(Date.now() / 1000 / 3600) * 3600;
    env.DB.q('INSERT INTO rate_counters (bucket, window_start, count) VALUES (?1, ?2, ?3)', 'ip:203.0.113.9', windowStart, AUTH_REQUEST_PER_IP_PER_HOUR);
    const spoofed = post('/v1/auth/request', { email: 'xff@example.com' }, { extra: { 'X-Forwarded-For': '203.0.113.9' } });
    expect((await worker.fetch(spoofed, env)).status).toBe(200);
    expect(env.DB.q('SELECT count FROM rate_counters WHERE bucket = ?1 AND window_start = ?2', 'ip:unknown', windowStart)[0].count).toBe(1);
    const both = post('/v1/auth/request', { email: 'both@example.com' }, { extra: { 'CF-Connecting-IP': '198.51.100.2', 'X-Forwarded-For': '203.0.113.9' } });
    expect((await worker.fetch(both, env)).status).toBe(200);
    expect(env.DB.q('SELECT count FROM rate_counters WHERE bucket = ?1 AND window_start = ?2', 'ip:198.51.100.2', windowStart)[0].count).toBe(1);
    expect(env.DB.q('SELECT count FROM rate_counters WHERE bucket = ?1 AND window_start = ?2', 'ip:203.0.113.9', windowStart)[0].count).toBe(AUTH_REQUEST_PER_IP_PER_HOUR);
  });
});

describe('WEB-AUTH-1 confirm page format / XSS / form POST', () => {
  it('escapes HTML attribute metacharacters even after the format check', () => {
    expect(escapeHtmlAttr(`&<>"'`)).toBe('&amp;&lt;&gt;&quot;&#39;');
  });

  it('a </script>-style token is 400 at GET and at POST and is never rendered or consumed', async () => {
    const xss = '</script><script>alert(1)</script>';
    const get = await worker.fetch(new Request(`https://api.rathor.ai/v1/auth/verify?token=${encodeURIComponent(xss)}`), env);
    expect(get.status).toBe(400);
    const getBody = await get.text();
    expect(getBody).not.toMatch(/<script/i);
    expect(getBody).not.toContain(xss);
    expect(JSON.parse(getBody)).toEqual({ error: 'invalid_token' });

    const posted = await worker.fetch(post('/v1/auth/verify', { token: xss }), env);
    expect(posted.status).toBe(400);
    expect(await posted.json()).toEqual({ error: 'invalid_token' });
    const form = await worker.fetch(formPost(xss), env);
    expect(form.status).toBe(400);
    expect(await form.json()).toEqual({ error: 'invalid_token' });
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(0);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM auth_tokens')[0].n).toBe(0);
  });

  it('a wrong-length token is 400 at GET and at POST', async () => {
    const short = 'A'.repeat(AUTH_TOKEN_LENGTH - 1);
    const long = 'A'.repeat(AUTH_TOKEN_LENGTH + 1);
    for (const token of [short, long]) {
      const get = await worker.fetch(new Request(`https://api.rathor.ai/v1/auth/verify?token=${token}`), env);
      expect(get.status).toBe(400);
      expect(await get.json()).toEqual({ error: 'invalid_token' });
      const posted = await worker.fetch(post('/v1/auth/verify', { token }), env);
      expect(posted.status).toBe(400);
      expect(await posted.json()).toEqual({ error: 'invalid_token' });
    }
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(0);
  });

  it('confirm page sets CSP, X-Frame-Options, Referrer-Policy and has no script tag', async () => {
    const token = randomToken();
    const get = await worker.fetch(new Request(`https://api.rathor.ai/v1/auth/verify?token=${token}`), env);
    expect(get.status).toBe(200);
    expectVerifyPageHeaders(get);
    expect(get.headers.get('Content-Security-Policy')).toContain(`form-action 'self' ${env.ALLOWED_ORIGIN}`);
    const html = await get.text();
    expect(html).not.toMatch(/<script/i);
    expect(html).not.toMatch(/style=/i);
    expect(html).toContain(`value="${token}"`);
  });

  it('form POST with a bad or missing Origin is 403 and does not consume', async () => {
    const f = stripeCheckoutFetch();
    await worker.fetch(post('/v1/auth/request', { email: 'form-origin@example.com' }), env);
    const { token } = extractToken(f);
    const hash = await sha256Hex(token);
    const bad = await worker.fetch(formPost(token, { origin: 'https://evil.example' }), env);
    expect(bad.status).toBe(403);
    expect(await bad.json()).toEqual({ error: 'origin_not_allowed' });
    const missing = await worker.fetch(formPost(token, { origin: null }), env);
    expect(missing.status).toBe(403);
    expect(env.DB.q('SELECT consumed_at FROM auth_tokens WHERE token_hash = ?1', hash)[0].consumed_at).toBeNull();
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(0);
  });

  it('form POST with Origin null is 403 and does not consume', async () => {
    const f = stripeCheckoutFetch();
    await worker.fetch(post('/v1/auth/request', { email: 'form-null-origin@example.com' }), env);
    const { token } = extractToken(f);
    const hash = await sha256Hex(token);
    const nullOrigin = await worker.fetch(formPost(token, { origin: 'null' }), env);
    expect(nullOrigin.status).toBe(403);
    expect(await nullOrigin.json()).toEqual({ error: 'origin_not_allowed' });
    expect(env.DB.q('SELECT consumed_at FROM auth_tokens WHERE token_hash = ?1', hash)[0].consumed_at).toBeNull();
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(0);
  });

  it('valid form POST consumes once, 303s to https://rathor.ai, and sets the host-only cookie', async () => {
    const f = stripeCheckoutFetch();
    await worker.fetch(post('/v1/auth/request', { email: 'form-ok@example.com' }), env);
    const { token } = extractToken(f);
    const r = await worker.fetch(formPost(token), env);
    expect(r.status).toBe(303);
    expect(r.headers.get('Location')).toBe(O);
    const cookie = r.headers.get('Set-Cookie');
    expect(cookie).toMatch(new RegExp(`^${SESSION_COOKIE_NAME}=`));
    expect(cookie).toMatch(/HttpOnly/);
    expect(cookie).toMatch(/Secure/);
    expect(cookie).toMatch(/SameSite=Lax/);
    expect(cookie).toMatch(/Path=\//);
    expect(cookie).not.toMatch(/Domain=/i);
    expect(cookie).toMatch(/Max-Age=2592000/);
    const raw = cookie.split(';')[0].split('=')[1];
    expect(env.DB.q('SELECT session_hash FROM sessions')[0].session_hash).toBe(await sha256Hex(raw));
    expect(env.DB.q('SELECT consumed_at FROM auth_tokens')[0].consumed_at).toBeTruthy();

    const again = await worker.fetch(formPost(token), env);
    expect(again.status).toBe(400);
    expectVerifyPageHeaders(again);
    const html = await again.text();
    expect(html).toContain(VERIFY_FAIL_MESSAGE);
    expect(html).not.toMatch(/<script/i);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(1);
  });

  it('form-path expired or used token renders the failure page with security headers', async () => {
    const token = randomToken();
    const past = Math.floor(Date.now() / 1000) - 10;
    env.DB.q('INSERT INTO auth_tokens (token_hash, email, expires_at) VALUES (?1, ?2, ?3)', await sha256Hex(token), 'form-late@example.com', past);
    const r = await worker.fetch(formPost(token), env);
    expect(r.status).toBe(400);
    expectVerifyPageHeaders(r);
    const html = await r.text();
    expect(html).toContain('This sign-in link has expired or was already used. Please request a new one.');
    expect(html).not.toMatch(/<script/i);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(0);
  });
});

describe('SPROOF-FIX-ORIGIN confirm Referrer-Policy', () => {
  it('Referrer-Policy is same-origin and is not no-referrer', async () => {
    const token = randomToken();
    const get = await worker.fetch(new Request(`https://api.rathor.ai/v1/auth/verify?token=${token}`), env);
    expect(get.status).toBe(200);
    expect(get.headers.get('Referrer-Policy')).not.toBe('no-referrer');
    expect(get.headers.get('Referrer-Policy')).toBe(VERIFY_PAGE_REFERRER_POLICY);
    expect(VERIFY_PAGE_REFERRER_POLICY).toBe('same-origin');
    expect(get.headers.get('X-Frame-Options')).toBe('DENY');
    expect(get.headers.get('Cache-Control')).toBe('no-store');
    const csp = get.headers.get('Content-Security-Policy');
    expect(csp).toContain(`form-action 'self' ${env.ALLOWED_ORIGIN}`);
    expect(csp).toContain("script-src 'none'");
  });

  it('confirm form POST with a real matching Origin succeeds', async () => {
    const f = stripeCheckoutFetch();
    await worker.fetch(post('/v1/auth/request', { email: 'sproof-origin-ok@example.com' }), env);
    const { token } = extractToken(f);
    const r = await worker.fetch(formPost(token, { origin: 'https://api.rathor.ai' }), env);
    expect(r.status).toBe(303);
    expect(r.headers.get('Location')).toBe(O);
    expect(env.DB.q('SELECT consumed_at FROM auth_tokens')[0].consumed_at).toBeTruthy();
  });

  it('confirm form POST with Origin null still returns 403', async () => {
    const f = stripeCheckoutFetch();
    await worker.fetch(post('/v1/auth/request', { email: 'sproof-origin-null@example.com' }), env);
    const { token } = extractToken(f);
    const r = await worker.fetch(formPost(token, { origin: 'null' }), env);
    expect(r.status).toBe(403);
    expect(await r.json()).toEqual({ error: 'origin_not_allowed' });
    expect(env.DB.q('SELECT consumed_at FROM auth_tokens')[0].consumed_at).toBeNull();
  });
});

describe('WEB-AUTH-1 CSRF / Origin / CORS / logout', () => {
  it('state-changing session routes refuse bad Origin or missing CSRF with 403', async () => {
    const cookie = await putSession();
    for (const path of ['/v1/checkout', '/v1/auth/logout', '/v1/keys', '/v1/keys/revoke']) {
      const body = path === '/v1/checkout' ? { price: 'pack_10' } : {};
      const badOrigin = await worker.fetch(post(path, body, { cookie, origin: 'https://evil.example' }), env);
      expect(badOrigin.status).toBe(403);
      expect(await badOrigin.json()).toEqual({ error: 'origin_not_allowed' });
      const noCsrf = await worker.fetch(post(path, body, { cookie, csrf: null }), env);
      expect(noCsrf.status).toBe(403);
      expect(await noCsrf.json()).toEqual({ error: 'csrf_required' });
    }
  });

  it('CORS allows credentials only from https://rathor.ai', async () => {
    const ok = await worker.fetch(new Request('https://api.rathor.ai/health', { headers: { Origin: O } }), env);
    expect(ok.headers.get('Access-Control-Allow-Origin')).toBe(O);
    expect(ok.headers.get('Access-Control-Allow-Credentials')).toBe('true');
    expect(ok.headers.get('Access-Control-Allow-Headers')).toContain(CSRF_HEADER);
    const pre = await worker.fetch(new Request('https://api.rathor.ai/v1/checkout', { method: 'OPTIONS', headers: { Origin: O } }), env);
    expect(pre.status).toBe(204);
    expect(pre.headers.get('Access-Control-Allow-Credentials')).toBe('true');
    const other = await worker.fetch(new Request('https://api.rathor.ai/health', { headers: { Origin: 'https://evil.example' } }), env);
    expect(other.headers.get('Access-Control-Allow-Origin')).toBeNull();
    expect(other.headers.get('Access-Control-Allow-Credentials')).toBeNull();
  });

  it('logout deletes the session and clears the cookie', async () => {
    const cookie = await putSession();
    const r = await worker.fetch(post('/v1/auth/logout', {}, { cookie }), env);
    expect(r.status).toBe(200);
    expect(r.headers.get('Set-Cookie')).toMatch(/Max-Age=0/);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM sessions')[0].n).toBe(0);
    okChatFetch();
    expect((await worker.fetch(post('/v1/chat', { messages: [{ role: 'user', content: 'hi' }] }, { cookie }), env)).status).toBe(401);
  });
});

describe('WEB-AUTH-1 checkout', () => {
  it('maps price keys, sets client_reference_id, uses payment or subscription, returns only the URL, ignores client amounts', async () => {
    const cookie = await putSession();
    const seen = [];
    stripeCheckoutFetch((params) => seen.push(Object.fromEntries(params)));
    const pack = await worker.fetch(post('/v1/checkout', { price: 'pack_10', amount: 1, credits: 999999, price_id: 'price_evil' }, { cookie }), env);
    expect(pack.status).toBe(200);
    expect(await pack.json()).toEqual({ url: CHECKOUT_URL });
    expect(seen[0]).toMatchObject({
      mode: 'payment',
      client_reference_id: 'u1',
      'line_items[0][price]': 'price_test_p10',
      'line_items[0][quantity]': '1',
    });
    expect(seen[0]).not.toHaveProperty('amount');
    expect(JSON.stringify(seen[0])).not.toContain('999999');
    expect(seen[0]['line_items[0][price]']).not.toBe('price_evil');

    const plan = await worker.fetch(post('/v1/checkout', { price: 'plan_monthly' }, { cookie }), env);
    expect(plan.status).toBe(200);
    expect(seen[1].mode).toBe('subscription');
    expect(seen[1]['line_items[0][price]']).toBe('price_test_plan');
    expect(seen[1].client_reference_id).toBe('u1');
  });

  it('refuses unset price ids and unknown keys; trial is refused once redeemed', async () => {
    const cookie = await putSession();
    stripeCheckoutFetch();
    expect((await worker.fetch(post('/v1/checkout', { price: 'nope' }, { cookie }), env)).status).toBe(400);
    env.STRIPE_PRICE_PACK_25 = '';
    const unset = await worker.fetch(post('/v1/checkout', { price: 'pack_25' }, { cookie }), env);
    expect(unset.status).toBe(503);
    expect(await unset.json()).toEqual({ error: 'price_not_configured' });
    env.DB.q("INSERT INTO trial_redemptions (account_id, stripe_event_id) VALUES ('u1', 'evt_prior')");
    const trial = await worker.fetch(post('/v1/checkout', { price: 'trial' }, { cookie }), env);
    expect(trial.status).toBe(409);
    expect(await trial.json()).toEqual({ error: 'trial_already_used' });
  });

  it('respects LIVE_MODE key/mode guard before calling Stripe', async () => {
    const cookie = await putSession();
    const f = stripeCheckoutFetch();
    env.LIVE_MODE = 'true';
    env.STRIPE_SECRET_KEY = SK_TEST;
    const r = await worker.fetch(post('/v1/checkout', { price: 'pack_10' }, { cookie }), env);
    expect(r.status).toBe(500);
    expect(await r.json()).toMatchObject({ error: 'stripe_key_mode_mismatch' });
    expect(f.mock.calls.filter((c) => String(c[0]).includes('checkout/sessions'))).toHaveLength(0);
    env.STRIPE_SECRET_KEY = SK_LIVE;
    const ok = await worker.fetch(post('/v1/checkout', { price: 'pack_10' }, { cookie }), env);
    expect(ok.status).toBe(200);
  });
});

describe('WEB-AUTH-1 session chat and nits', () => {
  it('session-authed /v1/chat charges the same ledger as a Bearer key', async () => {
    const cookie = await putSession();
    okChatFetch({ prompt_tokens: 1000, completion_tokens: 500 });
    const r = await worker.fetch(post('/v1/chat', { model: 'grok-4.7', messages: [{ role: 'user', content: 'hi' }] }, { cookie }), env);
    expect(r.status).toBe(200);
    const j = await r.json();
    expect(j.rathor.credits_charged).toBe(8);
    expect(bal()).toBe(992);
    expect(env.DB.q("SELECT delta FROM ledger WHERE reason='chat'")[0].delta).toBe(-8);
  });

  it('cookie-authenticated /v1/chat without CSRF is 403', async () => {
    const cookie = await putSession();
    const f = okChatFetch();
    const r = await worker.fetch(post('/v1/chat', { messages: [{ role: 'user', content: 'hi' }] }, { cookie, csrf: null }), env);
    expect(r.status).toBe(403);
    expect(await r.json()).toEqual({ error: 'csrf_required' });
    expect(f).not.toHaveBeenCalled();
    expect(bal()).toBe(1000);
  });

  it('Bearer-authenticated /v1/chat does not require CSRF', async () => {
    okChatFetch();
    const r = await worker.fetch(post('/v1/chat', { messages: [{ role: 'user', content: 'hi' }] }, { extra: { Authorization: `Bearer ${KEY}` }, csrf: null }), env);
    expect(r.status).toBe(200);
  });

  it('chat messages over 64 KB are 413', async () => {
    const cookie = await putSession();
    const f = okChatFetch();
    const big = 'x'.repeat(CHAT_MESSAGES_MAX_BYTES);
    const r = await worker.fetch(post('/v1/chat', { messages: [{ role: 'user', content: big }] }, { cookie }), env);
    expect(r.status).toBe(413);
    expect(await r.json()).toMatchObject({ error: 'payload_too_large', max_bytes: CHAT_MESSAGES_MAX_BYTES });
    expect(f).not.toHaveBeenCalled();
    expect(bal()).toBe(1000);
  });

  it('chat without a Stripe key is not 500', async () => {
    delete env.STRIPE_SECRET_KEY;
    okChatFetch();
    const r = await worker.fetch(post('/v1/chat', { messages: [{ role: 'user', content: 'hi' }] }, { extra: { Authorization: `Bearer ${KEY}` }, csrf: null }), env);
    expect(r.status).toBe(200);
  });
});

describe('WEB-AUTH-1 customer keys vs Stripe guard', () => {
  it('issues rth_ keys with Cache-Control no-store; a rth_ value never satisfies the Stripe key/mode guard', async () => {
    const cookie = await putSession();
    const r = await worker.fetch(post('/v1/keys', {}, { cookie }), env);
    expect(r.status).toBe(200);
    expect(r.headers.get('Cache-Control')).toBe('no-store');
    const { key } = await r.json();
    expect(key.startsWith(CUSTOMER_KEY_PREFIX_TEST)).toBe(true);
    expect(stripeKeyPrefixMode(key)).toBeNull();
    expect(stripeKeyMatchesLiveMode({ LIVE_MODE: 'false', STRIPE_SECRET_KEY: key })).toBe(false);
    expect(stripeKeyMatchesLiveMode({ LIVE_MODE: 'true', STRIPE_SECRET_KEY: key })).toBe(false);
    const built = ['rth', 'live', 'PLACEHOLDER'].join('_');
    expect(stripeKeyPrefixMode(built)).toBeNull();
    expect(stripeKeyPrefixMode(SK_TEST)).toBe('test');
    expect(stripeKeyPrefixMode(['rk', 'live', 'PLACEHOLDER'].join('_'))).toBe('live');
  });
});

describe('WEB-AUTH-1 client key scan', () => {
  it('client-facing HTML/JS has no secret key patterns', () => {
    const script = fileURLToPath(new URL('../scripts/scan-client-keys.mjs', import.meta.url));
    const r = spawnSync(process.execPath, [script], { encoding: 'utf8' });
    expect(r.status, r.stderr).toBe(0);
    expect(r.stdout).toMatch(/ok: scanned/);
  });
});
