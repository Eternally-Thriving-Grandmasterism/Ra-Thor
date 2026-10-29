// WEB-AUTH-1: magic-link tokens, hashed sessions, CSRF, rate limits.
// The browser holds only the session cookie. Raw tokens and session ids are never stored.
import { sha256Hex, constantTimeEqual } from './hash.js';
import {
  AUTH_TOKEN_TTL_SEC, SESSION_TTL_SEC, SESSION_COOKIE_NAME, CSRF_HEADER,
  AUTH_REQUEST_PER_EMAIL_PER_HOUR, AUTH_REQUEST_PER_IP_PER_HOUR,
  CUSTOMER_KEY_PREFIX_TEST, CUSTOMER_KEY_PREFIX_LIVE,
} from './config.js';
import { SIGNUP_GRANT_CREDITS } from './db.js';

export const json = (obj, status, headers = {}) =>
  new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json', ...headers } });

export const nowSec = (env) => (Number.isInteger(env?.NOW_SEC) ? env.NOW_SEC : Math.floor(Date.now() / 1000));

// 32 random bytes → base64url without padding. Length is ceil(32*8/6) = 43.
export const AUTH_TOKEN_BYTES = 32;
export const AUTH_TOKEN_LENGTH = Math.ceil((AUTH_TOKEN_BYTES * 8) / 6);
export const AUTH_TOKEN_RE = new RegExp(`^[A-Za-z0-9_-]{${AUTH_TOKEN_LENGTH}}$`);
export const isAuthToken = (s) => typeof s === 'string' && AUTH_TOKEN_RE.test(s);

export const VERIFY_FAIL_MESSAGE = 'This sign-in link has expired or was already used. Please request a new one.';

// Same source as the verify Origin check and the form-success 303 Location.
export function allowedOrigin(env) {
  return env?.ALLOWED_ORIGIN || 'https://rathor.ai';
}

// same-origin: the confirm form POSTs to relative /v1/auth/verify, so Chromium
// serializes a real Origin. Cross-origin requests omit Referer, so the magic-link
// token in the GET query never reaches a foreign origin as a referrer.
// no-referrer made Chromium send Origin: null; verifyOriginGuard correctly 403s
// that. Origin: null stays 403 — do not special-case it.
export const VERIFY_PAGE_REFERRER_POLICY = 'same-origin';

// form-action includes ALLOWED_ORIGIN so Chromium will follow the 303 to that origin.
export function verifyPageHeaders(env) {
  const dest = allowedOrigin(env);
  return {
    'Content-Type': 'text/html; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': VERIFY_PAGE_REFERRER_POLICY,
    'Content-Security-Policy': `default-src 'none'; script-src 'none'; connect-src 'none'; form-action 'self' ${dest}; frame-ancestors 'none'; base-uri 'none'`,
  };
}

export function randomToken(bytes = AUTH_TOKEN_BYTES) {
  const a = new Uint8Array(bytes);
  crypto.getRandomValues(a);
  let bin = '';
  for (const b of a) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

export function normalizeEmail(raw) {
  if (typeof raw !== 'string') return null;
  const e = raw.trim().toLowerCase();
  if (e.length < 3 || e.length > 254) return null;
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e)) return null;
  if (e.includes('..')) return null;
  return e;
}

// Cloudflare sets CF-Connecting-IP. No X-Forwarded-For fallback (clients can spoof it).
// Missing header shares one 'unknown' bucket.
export function clientIp(req) {
  const cf = req.headers.get('CF-Connecting-IP');
  return cf && cf.trim() ? cf.trim() : 'unknown';
}

export function parseCookie(req, name = SESSION_COOKIE_NAME) {
  const raw = req.headers.get('Cookie') || '';
  for (const part of raw.split(';')) {
    const i = part.indexOf('=');
    if (i < 0) continue;
    if (part.slice(0, i).trim() === name) return part.slice(i + 1).trim();
  }
  return null;
}

// Host-only on api.rathor.ai (no Domain attribute). Same-site credentialed
// fetch from https://rathor.ai still sends it because the request host is api.rathor.ai.
export function sessionCookie(raw, env, { clear = false } = {}) {
  const maxAge = clear ? 0 : SESSION_TTL_SEC;
  const value = clear ? '' : raw;
  return `${SESSION_COOKIE_NAME}=${value}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`;
}

// Custom-header + Origin CSRF. Browsers will not send X-Rathor-CSRF on a
// simple cross-site form POST (it triggers a CORS preflight). Combined with
// Origin === ALLOWED_ORIGIN this refuses foreign-site state changes.
export function csrfGuard(req, env) {
  const allowed = env.ALLOWED_ORIGIN || 'https://rathor.ai';
  if (req.headers.get('Origin') !== allowed) return { error: 'origin_not_allowed', status: 403 };
  const token = req.headers.get(CSRF_HEADER);
  if (!token) return { error: 'csrf_required', status: 403 };
  return null;
}

// POST /v1/auth/verify: exact Origin must be ALLOWED_ORIGIN (rathor.ai UI) or
// the API origin serving the GET confirm page (form posts from there).
export function verifyOriginGuard(req, env) {
  const origin = req.headers.get('Origin');
  const allowed = allowedOrigin(env);
  const apiOrigin = new URL(req.url).origin;
  if (origin !== allowed && origin !== apiOrigin) return { error: 'origin_not_allowed', status: 403 };
  return null;
}

// JSON callers send X-Rathor-CSRF. The no-JS confirm form sends a `csrf` field.
export function verifyCsrfGuard(req, formCsrf) {
  if (req.headers.get(CSRF_HEADER) || formCsrf) return null;
  return { error: 'csrf_required', status: 403 };
}

export async function authenticateSession(req, env) {
  const raw = parseCookie(req);
  if (!raw || raw.length < 20 || raw.length > 200) return null;
  const hash = await sha256Hex(raw);
  const row = await env.DB.prepare(
    'SELECT session_hash, account_id FROM sessions WHERE session_hash = ?1 AND revoked = 0 AND expires_at > ?2',
  ).bind(hash, nowSec(env)).first();
  if (!row || !constantTimeEqual(row.session_hash, hash)) return null;
  return row.account_id;
}

export async function requireSessionCsrf(req, env, h) {
  const csrf = csrfGuard(req, env);
  if (csrf) return { response: json({ error: csrf.error }, csrf.status, h) };
  const user = await authenticateSession(req, env);
  if (!user) return { response: json({ error: 'unauthorized' }, 401, h) };
  return { user };
}

async function rateHit(db, bucket, windowStart, limit) {
  await db.prepare(
    `INSERT INTO rate_counters (bucket, window_start, count) VALUES (?1, ?2, 1)
     ON CONFLICT(bucket, window_start) DO UPDATE SET count = count + 1`,
  ).bind(bucket, windowStart).run();
  const row = await db.prepare('SELECT count FROM rate_counters WHERE bucket = ?1 AND window_start = ?2')
    .bind(bucket, windowStart).first();
  return (row?.count ?? 0) > limit;
}

export async function findOrCreateAccountByEmail(db, email) {
  const existing = await db.prepare('SELECT id FROM accounts WHERE email = ?1').bind(email).first();
  if (existing) return { id: existing.id, created: false };
  const id = crypto.randomUUID();
  try {
    await db.prepare('INSERT INTO accounts (id, email, balance) VALUES (?1, ?2, ?3)')
      .bind(id, email, SIGNUP_GRANT_CREDITS).run();
    return { id, created: true };
  } catch (e) {
    if (!/UNIQUE/i.test(String(e?.message || e))) throw e;
    const again = await db.prepare('SELECT id FROM accounts WHERE email = ?1').bind(email).first();
    if (!again) throw e;
    return { id: again.id, created: false };
  }
}

export async function consumeAuthToken(db, tokenHash, now) {
  const r = await db.prepare(
    'UPDATE auth_tokens SET consumed_at = ?2 WHERE token_hash = ?1 AND consumed_at IS NULL AND expires_at > ?2',
  ).bind(tokenHash, now).run();
  if ((r.meta?.changes ?? 0) !== 1) return null;
  return db.prepare('SELECT email FROM auth_tokens WHERE token_hash = ?1').bind(tokenHash).first();
}

export async function createSessionRow(db, sessionHash, accountId, expiresAt) {
  await db.prepare('INSERT INTO sessions (session_hash, account_id, expires_at) VALUES (?1, ?2, ?3)')
    .bind(sessionHash, accountId, expiresAt).run();
}

export async function deleteSession(db, sessionHash) {
  await db.prepare('DELETE FROM sessions WHERE session_hash = ?1').bind(sessionHash).run();
}

// POST /v1/auth/request {email}. Always the same 200 body. Never reveals whether
// the email already has an account. Email send is best-effort.
export async function requestMagicLink(req, env, h) {
  const allowed = env.ALLOWED_ORIGIN || 'https://rathor.ai';
  if (req.headers.get('Origin') !== allowed) return json({ error: 'origin_not_allowed' }, 403, h);
  let body;
  try { body = await req.json(); } catch { return json({ error: 'bad_json' }, 400, h); }
  const email = normalizeEmail(body?.email);
  if (!email) return json({ error: 'bad_email' }, 400, h);

  const now = nowSec(env);
  const windowStart = Math.floor(now / 3600) * 3600;
  const emailBucket = `email:${await sha256Hex(email)}`;
  const ipBucket = `ip:${clientIp(req)}`;
  if (await rateHit(env.DB, emailBucket, windowStart, AUTH_REQUEST_PER_EMAIL_PER_HOUR)
    || await rateHit(env.DB, ipBucket, windowStart, AUTH_REQUEST_PER_IP_PER_HOUR)) {
    return json({ error: 'rate_limited' }, 429, h);
  }

  const token = randomToken(32);
  const hash = await sha256Hex(token);
  await env.DB.prepare('INSERT INTO auth_tokens (token_hash, email, expires_at) VALUES (?1, ?2, ?3)')
    .bind(hash, email, now + AUTH_TOKEN_TTL_SEC).run();

  if (env.RESEND_API_KEY) {
    const origin = new URL(req.url).origin;
    const link = `${origin}/v1/auth/verify?token=${encodeURIComponent(token)}`;
    try {
      await fetch('https://api.resend.com/emails', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.RESEND_API_KEY}` },
        body: JSON.stringify({
          from: 'Ra-Thor <info@rathor.ai>',
          to: [email],
          subject: 'Sign in to Ra-Thor',
          text: `Click this link to sign in (expires in 15 minutes):\n\n${link}\n\nIf you did not request this, ignore this email.`,
        }),
      });
    } catch { /* best effort; same 200 either way */ }
  }
  return json({ ok: true }, 200, h);
}

// Escape for an HTML attribute value even after AUTH_TOKEN_RE (defense in depth).
export function escapeHtmlAttr(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function verifyPage(html, status, h, env) {
  return new Response(html, { status, headers: { ...h, ...verifyPageHeaders(env) } });
}

function isFormUrlEncoded(req) {
  const ct = (req.headers.get('Content-Type') || '').split(';')[0].trim().toLowerCase();
  return ct === 'application/x-www-form-urlencoded';
}

// GET /v1/auth/verify: confirm page only. Never consumes the token or sets a cookie.
// No JavaScript: a plain form POST keeps the token out of any script context.
export function confirmPageHtml(token) {
  const safe = escapeHtmlAttr(token);
  // csrf field value is the constant 1; the exact-Origin check does the real protection.
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Sign in to Ra-Thor</title></head><body>
<p>Confirm sign-in to Ra-Thor.</p>
<form method="POST" action="/v1/auth/verify">
<input type="hidden" name="token" value="${safe}">
<input type="hidden" name="csrf" value="1">
<button type="submit">Sign in</button>
</form>
</body></html>`;
}

export function verifyFailPageHtml() {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Sign in to Ra-Thor</title></head><body>
<p>${VERIFY_FAIL_MESSAGE}</p>
</body></html>`;
}

export function verifyConfirmPage(req, env, h) {
  const token = new URL(req.url).searchParams.get('token');
  if (!isAuthToken(token)) return json({ error: 'invalid_token' }, 400, h);
  return verifyPage(confirmPageHtml(token), 200, h, env);
}

// POST /v1/auth/verify: JSON + X-Rathor-CSRF, or form-urlencoded + csrf field.
// Exact Origin is required either way. Form success is 303 to ALLOWED_ORIGIN.
export async function verifyMagicLink(req, env, h) {
  const originFail = verifyOriginGuard(req, env);
  if (originFail) return json({ error: originFail.error }, originFail.status, h);

  const form = isFormUrlEncoded(req);
  let token = null;
  let formCsrf = null;
  if (form) {
    const params = new URLSearchParams(await req.text());
    token = params.get('token');
    formCsrf = params.get('csrf');
  } else {
    try {
      const body = await req.json();
      token = typeof body?.token === 'string' ? body.token : null;
    } catch { return json({ error: 'bad_json' }, 400, h); }
  }

  const csrf = verifyCsrfGuard(req, formCsrf);
  if (csrf) return json({ error: csrf.error }, csrf.status, h);
  if (!isAuthToken(token)) return json({ error: 'invalid_token' }, 400, h);

  const now = nowSec(env);
  const consumed = await consumeAuthToken(env.DB, await sha256Hex(token), now);
  if (!consumed?.email) {
    if (form) return verifyPage(verifyFailPageHtml(), 400, h, env);
    return json({ error: 'invalid_token', message: VERIFY_FAIL_MESSAGE }, 400, h);
  }
  const acct = await findOrCreateAccountByEmail(env.DB, consumed.email);
  const raw = randomToken();
  await createSessionRow(env.DB, await sha256Hex(raw), acct.id, now + SESSION_TTL_SEC);
  const cookie = sessionCookie(raw, env);
  if (form) {
    const dest = allowedOrigin(env);
    return new Response(null, {
      status: 303,
      headers: { ...h, Location: dest, 'Set-Cookie': cookie, 'Cache-Control': 'no-store' },
    });
  }
  return json({ ok: true }, 200, { ...h, 'Set-Cookie': cookie });
}

export async function logout(req, env, h) {
  const gate = await requireSessionCsrf(req, env, h);
  if (gate.response) return gate.response;
  const raw = parseCookie(req);
  if (raw) await deleteSession(env.DB, await sha256Hex(raw));
  return json({ ok: true }, 200, { ...h, 'Set-Cookie': sessionCookie('', env, { clear: true }) });
}

function keyPrefix(env) {
  return String(env?.LIVE_MODE) === 'true' ? CUSTOMER_KEY_PREFIX_LIVE : CUSTOMER_KEY_PREFIX_TEST;
}

export async function issueKey(req, env, h) {
  const gate = await requireSessionCsrf(req, env, h);
  if (gate.response) return gate.response;
  const key = keyPrefix(env) + randomToken(32);
  const hash = await sha256Hex(key);
  await env.DB.prepare('INSERT INTO api_keys (key_hash, account_id) VALUES (?1, ?2)').bind(hash, gate.user).run();
  return json({ key }, 200, { ...h, 'Cache-Control': 'no-store' });
}

export async function revokeKeys(req, env, h) {
  const gate = await requireSessionCsrf(req, env, h);
  if (gate.response) return gate.response;
  await env.DB.prepare('UPDATE api_keys SET revoked = 1 WHERE account_id = ?1 AND revoked = 0').bind(gate.user).run();
  return json({ ok: true }, 200, h);
}
