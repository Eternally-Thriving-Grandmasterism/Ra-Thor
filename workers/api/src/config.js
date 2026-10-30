// Server-side request limits. Anything outside these is rejected with 400.
// Launch allowlist (PRICING-1): only models with base rates in src/credits.js.
export const ALLOWED_MODELS = Object.freeze(['grok-4.7', 'grok-4.3']);
export const DEFAULT_MODEL = 'grok-4.3';
// LIVE-GATE-1: output cap. validateChat clamps requested max_tokens to this and
// defaults to it when absent. placeHold uses this as the worst-case completion.
export const MAX_TOKENS_CAP = 4096;
export const DEFAULT_MAX_TOKENS = 4096;
// WEB-AUTH-1: JSON-encoded `messages` larger than this is 413. Documented in README.
export const CHAT_MESSAGES_MAX_BYTES = 64 * 1024;

export const AUTH_TOKEN_TTL_SEC = 15 * 60;
export const SESSION_TTL_SEC = 30 * 24 * 60 * 60;
export const SESSION_COOKIE_NAME = 'rathor_session';
export const CSRF_HEADER = 'X-Rathor-CSRF';
// Customer API keys (not Stripe). LIVE-GATE-1 stripeKeyPrefixMode matches only sk_/rk_.
export const CUSTOMER_KEY_PREFIX_TEST = ['rth', 'test', ''].join('_');
export const CUSTOMER_KEY_PREFIX_LIVE = ['rth', 'live', ''].join('_');
export const AUTH_REQUEST_PER_EMAIL_PER_HOUR = 5;
export const AUTH_REQUEST_PER_IP_PER_HOUR = 20;

// API-ABUSE-HARDEN-1: per-account limits (D1 rate_counters, fixed window). Over the limit is
// 429 {error:'rate_limited', retry_after}; nothing is charged and no ledger row is written.
export const RATE_WINDOW_SEC = 60;
export const CHAT_RATE_PER_ACCOUNT_PER_WINDOW = 20;
export const CREDITS_RATE_PER_ACCOUNT_PER_WINDOW = 60;
// xAI fetch timeout. XAI_TIMEOUT_MS is clamped to 60000-90000; unset or invalid -> 75000.
export const XAI_TIMEOUT_MS_DEFAULT = 75000;
export const XAI_TIMEOUT_MS_MIN = 60000;
export const XAI_TIMEOUT_MS_MAX = 90000;
export function xaiTimeoutMs(env) {
  const n = Number.parseInt(String(env?.XAI_TIMEOUT_MS ?? ''), 10);
  if (!Number.isInteger(n)) return XAI_TIMEOUT_MS_DEFAULT;
  return Math.min(XAI_TIMEOUT_MS_MAX, Math.max(XAI_TIMEOUT_MS_MIN, n));
}
// Backoff (ms) between settleHold retries after xAI usage has come back.
export const SETTLE_RETRY_BACKOFF_MS = Object.freeze([25, 100, 400]);
// mercy_blocks rows per window; past this only the aggregate counter (rate_counters) grows.
export const MERCY_BLOCK_WINDOW_SEC = 3600;
export const MERCY_BLOCK_ROWS_PER_WINDOW = 500;

// Fail-closed live/sandbox gate. Only the exact string "true" is live; unset,
// misspelled, or any other value is sandbox.
export const isLiveMode = (env) => String(env?.LIVE_MODE) === 'true';

// STRIPE-404-RETRY-1 exhaust: after this many seconds a transient Stripe
// fetch (404/429/5xx/network/bad JSON) is recorded once and answered 200.
// Unset, invalid, or under 3600 (including "0") → 262800 (~73h).
// SPROOF-FIX-DISPUTE-ORDER: production must outlast Stripe's 72h (259200s) webhook
// retry schedule plus a 1h margin: 259200 + 3600 = 262800.
// Sandbox wrangler.toml sets 21600 (~6h).
export const STRIPE_RETRY_EXHAUST_SECONDS_STRIPE_RETRY_WINDOW = 259200;
export const STRIPE_RETRY_EXHAUST_SECONDS_MARGIN = 3600;
export const STRIPE_RETRY_EXHAUST_SECONDS_DEFAULT = STRIPE_RETRY_EXHAUST_SECONDS_STRIPE_RETRY_WINDOW + STRIPE_RETRY_EXHAUST_SECONDS_MARGIN;
export const STRIPE_RETRY_EXHAUST_SECONDS_MIN = 3600;
export function stripeRetryExhaustSeconds(env) {
  const raw = env?.STRIPE_RETRY_EXHAUST_SECONDS;
  if (raw == null || raw === '') return STRIPE_RETRY_EXHAUST_SECONDS_DEFAULT;
  const n = Number.parseInt(String(raw), 10);
  if (!Number.isInteger(n) || n < STRIPE_RETRY_EXHAUST_SECONDS_MIN || String(raw).trim() !== String(n)) {
    return STRIPE_RETRY_EXHAUST_SECONDS_DEFAULT;
  }
  return n;
}

// Prefix-only classification. Cheap (startsWith). No secret material is logged.
export function stripeKeyPrefixMode(key) {
  if (typeof key !== 'string') return null;
  if (key.startsWith('sk_live_') || key.startsWith('rk_live_')) return 'live';
  if (key.startsWith('sk_test_') || key.startsWith('rk_test_')) return 'test';
  return null;
}

// LIVE_MODE true requires a live key; otherwise a test key. Mismatch or unknown
// prefix is false (caller refuses with 500 stripe_key_mode_mismatch).
export function stripeKeyMatchesLiveMode(env) {
  const want = isLiveMode(env) ? 'live' : 'test';
  return stripeKeyPrefixMode(env?.STRIPE_SECRET_KEY) === want;
}
