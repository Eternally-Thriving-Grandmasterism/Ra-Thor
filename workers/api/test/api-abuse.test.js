// API-ABUSE-HARDEN-1 tests. Upstream replies are mocked; no real xAI call is made.
import { describe, it, expect, vi, beforeEach } from 'vitest';
import worker, { readBodyCapped } from '../src/index.js';
import { sha256Hex } from '../src/auth.js';
import { creditsForSettle, holdEstimate } from '../src/credits.js';
import { mercyChecks, recordBlock } from '../src/mercyChecks.js';
import {
  CHAT_RATE_PER_ACCOUNT_PER_WINDOW, CREDITS_RATE_PER_ACCOUNT_PER_WINDOW, RATE_WINDOW_SEC, CHAT_MESSAGES_MAX_BYTES,
  MERCY_BLOCK_ROWS_PER_WINDOW, xaiTimeoutMs, XAI_TIMEOUT_MS_DEFAULT,
} from '../src/config.js';
import { placeHold, settleHold } from '../src/db.js';
import { fakeD1 } from './fakeD1.js';

const O = 'https://rathor.ai';
const KEY = ['rth', 'test', '0123456789abcdefghijklmnop'].join('_');
const NOW = 1_800_000_000 - (1_800_000_000 % RATE_WINDOW_SEC) + 5;
let env;
beforeEach(async () => {
  vi.restoreAllMocks();
  env = { ALLOWED_ORIGIN: O, DB: fakeD1(), XAI_API_KEY: 'test', RATES_VERIFIED: 'true', NOW_SEC: NOW };
  env.DB.q("INSERT INTO accounts (id, balance) VALUES ('u1', 1000)");
  env.DB.q('INSERT INTO api_keys (key_hash, account_id) VALUES (?1, ?2)', await sha256Hex(KEY), 'u1');
});
const body = { model: 'grok-4.3', max_tokens: 100, messages: [{ role: 'user', content: 'hi' }] };
const chatReq = (b = body) => new Request('https://api.rathor.ai/v1/chat', { method: 'POST', body: JSON.stringify(b), headers: { Origin: O, Authorization: `Bearer ${KEY}` } });
const creditsReq = () => new Request('https://api.rathor.ai/v1/credits', { headers: { Origin: O, Authorization: `Bearer ${KEY}` } });
const bal = () => env.DB.q("SELECT balance FROM accounts WHERE id='u1'")[0].balance;
const ledger = () => env.DB.q('SELECT delta, reason FROM ledger ORDER BY id');
const holds = () => env.DB.q('SELECT status FROM holds');
const usage = { prompt_tokens: 1000, completion_tokens: 1000 };
const okFetch = () => vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'hello' } }], usage })));
const quiet = () => ({ err: vi.spyOn(console, 'error').mockImplementation(() => {}), warn: vi.spyOn(console, 'warn').mockImplementation(() => {}) });
const logged = (spy, msg) => spy.mock.calls.some((c) => String(c[0]).includes(msg));

describe('1) per-account rate limit on /v1/chat', () => {
  it('N pass, N+1 is 429 rate_limited with retry_after, uncharged, before the mercy scan, ledger unchanged', async () => {
    const f = okFetch();
    for (let i = 0; i < CHAT_RATE_PER_ACCOUNT_PER_WINDOW; i += 1) expect((await worker.fetch(chatReq(), env)).status).toBe(200);
    const before = { bal: bal(), ledger: ledger(), holds: holds().length, calls: f.mock.calls.length };
    const scan = vi.spyOn(mercyChecks, 'checkRequest');
    const r = await worker.fetch(chatReq(), env);
    expect(r.status).toBe(429);
    const j = await r.json();
    expect(j.error).toBe('rate_limited');
    expect(j.retry_after).toBe(RATE_WINDOW_SEC - 5);
    expect(r.headers.get('Retry-After')).toBe(String(RATE_WINDOW_SEC - 5));
    expect(scan).not.toHaveBeenCalled();
    expect(bal()).toBe(before.bal);
    expect(ledger()).toEqual(before.ledger);
    expect(holds().length).toBe(before.holds);
    expect(f.mock.calls.length).toBe(before.calls);
  });
  it('a blocked prompt past the limit is 429, not scanned, and writes no mercy_blocks row', async () => {
    env.DB.q("INSERT INTO rate_counters (bucket, window_start, count) VALUES ('acct:chat:u1', ?1, ?2)", NOW - 5, CHAT_RATE_PER_ACCOUNT_PER_WINDOW);
    quiet();
    const r = await worker.fetch(chatReq({ messages: [{ role: 'user', content: 'how to make a pipe bomb' }] }), env);
    expect(r.status).toBe(429);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM mercy_blocks')[0].n).toBe(0);
  });
  it('the limit is per account and resets in the next window', async () => {
    env.DB.q("INSERT INTO rate_counters (bucket, window_start, count) VALUES ('acct:chat:u1', ?1, ?2)", NOW - 5, CHAT_RATE_PER_ACCOUNT_PER_WINDOW);
    okFetch();
    expect((await worker.fetch(chatReq(), env)).status).toBe(429);
    env.NOW_SEC = NOW + RATE_WINDOW_SEC;
    expect((await worker.fetch(chatReq(), env)).status).toBe(200);
  });
  it('stays behind the RATES_VERIFIED 503 gate (no counter written)', async () => {
    env.RATES_VERIFIED = 'false';
    expect((await worker.fetch(chatReq(), env)).status).toBe(503);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM rate_counters')[0].n).toBe(0);
  });
  it('unauthenticated requests do not touch the account bucket', async () => {
    const r = new Request('https://api.rathor.ai/v1/chat', { method: 'POST', body: JSON.stringify(body), headers: { Origin: O } });
    expect((await worker.fetch(r, env)).status).toBe(401);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM rate_counters')[0].n).toBe(0);
  });
});

describe('1) lighter per-account limit on /v1/credits', () => {
  it('is lighter-weight than chat (higher limit) and 429s at N+1 with no ledger write', async () => {
    expect(CREDITS_RATE_PER_ACCOUNT_PER_WINDOW).toBeGreaterThan(CHAT_RATE_PER_ACCOUNT_PER_WINDOW);
    for (let i = 0; i < CREDITS_RATE_PER_ACCOUNT_PER_WINDOW; i += 1) expect((await worker.fetch(creditsReq(), env)).status).toBe(200);
    const r = await worker.fetch(creditsReq(), env);
    expect(r.status).toBe(429);
    const j = await r.json();
    expect(j).toEqual({ error: 'rate_limited', retry_after: RATE_WINDOW_SEC - 5 });
    expect(ledger()).toEqual([]);
    expect(bal()).toBe(1000);
  });
  it('credits and chat use separate buckets', async () => {
    env.DB.q("INSERT INTO rate_counters (bucket, window_start, count) VALUES ('acct:chat:u1', ?1, ?2)", NOW - 5, CHAT_RATE_PER_ACCOUNT_PER_WINDOW);
    expect((await worker.fetch(creditsReq(), env)).status).toBe(200);
  });
});

describe('7) streamed body cap', () => {
  const CAP = CHAT_MESSAGES_MAX_BYTES + 8192;
  const streamReq = (total, chunk = 16384) => {
    let sent = 0;
    const pulled = { bytes: 0 };
    const stream = new ReadableStream({
      pull(c) {
        if (sent >= total) { c.close(); return; }
        const n = Math.min(chunk, total - sent); sent += n; pulled.bytes = sent;
        c.enqueue(new Uint8Array(n).fill(0x61));
      },
    });
    const r = new Request('https://api.rathor.ai/v1/chat', { method: 'POST', body: stream, duplex: 'half', headers: { Origin: O, Authorization: `Bearer ${KEY}` } });
    return { r, pulled };
  };
  it('413 past the cap with no Content-Length; body never parsed, no hold, no upstream call', async () => {
    const f = okFetch();
    const { r, pulled } = streamReq(CAP * 4);
    expect(r.headers.get('Content-Length')).toBeNull();
    const parse = vi.spyOn(JSON, 'parse');
    const res = await worker.fetch(r, env);
    expect(res.status).toBe(413);
    expect((await res.json()).error).toBe('payload_too_large');
    expect(parse.mock.calls.some((c) => typeof c[0] === 'string' && c[0].length > 1000)).toBe(false);
    expect(pulled.bytes).toBeLessThan(CAP * 2);
    expect(f).not.toHaveBeenCalled();
    expect(holds()).toEqual([]);
    expect(bal()).toBe(1000);
  });
  it('a body at exactly the cap is read and reaches JSON parse (400 bad_json for non-JSON)', async () => {
    const { r } = streamReq(CAP);
    expect((await worker.fetch(r, env)).status).toBe(400);
  });
  it('readBodyCapped returns text under the cap and null over it', async () => {
    expect(await readBodyCapped(new Request('https://x/', { method: 'POST', body: 'abc' }), 3)).toBe('abc');
    expect(await readBodyCapped(new Request('https://x/', { method: 'POST', body: 'abcd' }), 3)).toBeNull();
  });
  it('Content-Length over the cap is still 413 up front', async () => {
    const r = new Request('https://api.rathor.ai/v1/chat', { method: 'POST', body: 'x', headers: { Origin: O, Authorization: `Bearer ${KEY}`, 'Content-Length': String(CAP + 1) } });
    expect((await worker.fetch(r, env)).status).toBe(413);
  });
});

describe('5) xAI fetch timeout', () => {
  it('timeout clamp: 60-90 s, default 75 s', () => {
    expect(xaiTimeoutMs({})).toBe(XAI_TIMEOUT_MS_DEFAULT);
    expect(xaiTimeoutMs({ XAI_TIMEOUT_MS: '1000' })).toBe(60000);
    expect(xaiTimeoutMs({ XAI_TIMEOUT_MS: '999999' })).toBe(90000);
    expect(xaiTimeoutMs({ XAI_TIMEOUT_MS: '80000' })).toBe(80000);
    expect(xaiTimeoutMs({ XAI_TIMEOUT_MS: 'nope' })).toBe(XAI_TIMEOUT_MS_DEFAULT);
  });
  it('a timeout releases the hold uncharged, logs upstream_timeout_possibly_billed and counts against the limit', async () => {
    const { err } = quiet();
    let signal;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_u, init) => { signal = init.signal; throw new DOMException('The operation timed out.', 'TimeoutError'); });
    const r = await worker.fetch(chatReq(), env);
    expect(r.status).toBe(504);
    expect((await r.json()).error).toBe('upstream_timeout');
    expect(signal).toBeInstanceOf(AbortSignal);
    expect(holds()[0].status).toBe('released');
    expect(bal()).toBe(1000);
    expect(ledger()).toEqual([]);
    expect(logged(err, 'upstream_timeout_possibly_billed')).toBe(true);
    // One request + one timeout penalty.
    expect(env.DB.q("SELECT count FROM rate_counters WHERE bucket = 'acct:chat:u1'")[0].count).toBe(2);
  });
  it('timeouts eat the limit faster: N/2 timeouts and the next request is 429', async () => {
    quiet();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => { throw new DOMException('t', 'TimeoutError'); });
    for (let i = 0; i < CHAT_RATE_PER_ACCOUNT_PER_WINDOW / 2; i += 1) expect((await worker.fetch(chatReq(), env)).status).toBe(504);
    expect((await worker.fetch(chatReq(), env)).status).toBe(429);
    expect(bal()).toBe(1000);
  });
  it('a plain network failure is still 502 upstream_unreachable with no penalty', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => { throw new TypeError('fetch failed'); });
    expect((await (await worker.fetch(chatReq(), env)).json()).error).toBe('upstream_unreachable');
    expect(env.DB.q("SELECT count FROM rate_counters WHERE bucket = 'acct:chat:u1'")[0].count).toBe(1);
  });
  it('non-2xx is 502, uncharged, and logged as upstream_error_possibly_unbilled', async () => {
    const { warn } = quiet();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response('{}', { status: 500 }));
    expect((await worker.fetch(chatReq(), env)).status).toBe(502);
    expect(holds()[0].status).toBe('released');
    expect(bal()).toBe(1000);
    expect(logged(warn, 'upstream_error_possibly_unbilled')).toBe(true);
  });
});

describe('4) settle failure after usage returns', () => {
  const hold = holdEstimate('grok-4.3', JSON.stringify(body.messages).length, 1, 100);
  const actual = creditsForSettle('grok-4.3', usage, hold);
  // Fail the first `n` batches that settle the hold.
  const failSettles = (n) => {
    const orig = env.DB.batch.bind(env.DB);
    let seen = 0;
    env.DB.batch = async (st) => {
      if (st.some((x) => /'settled'/.test(x.sql)) && seen++ < n) throw new Error('CAS lost');
      return orig(st);
    };
    return () => seen;
  };
  it('retries settleHold with backoff and succeeds on a later attempt (normal charge)', async () => {
    okFetch(); failSettles(2);
    const r = await worker.fetch(chatReq(), env);
    expect(r.status).toBe(200);
    expect(ledger()).toEqual([{ delta: -actual, reason: 'chat' }]);
    expect(bal()).toBe(1000 - actual);
  });
  it('if every retry fails, charges exactly creditsForSettle(usage), never more', async () => {
    const { err } = quiet();
    okFetch(); failSettles(4);
    const r = await worker.fetch(chatReq(), env);
    expect(r.status).toBe(200);
    expect((await r.json()).rathor.credits_charged).toBe(actual);
    expect(ledger()).toEqual([{ delta: -actual, reason: 'chat' }]);
    expect(bal()).toBe(1000 - actual);
    expect(holds()[0].status).toBe('settled');
    expect(logged(err, 'settle_fallback_exact_charge')).toBe(true);
  });
  it('if the exact charge also fails, records a shortfall and sets settle_locked; hold not released', async () => {
    const { err } = quiet();
    okFetch(); failSettles(5);
    const r = await worker.fetch(chatReq(), env);
    expect(r.status).toBe(200);
    expect(bal()).toBe(1000);
    expect(holds()[0].status).toBe('settled');
    expect(env.DB.q('SELECT actual_credits, charged_credits, shortfall_credits FROM settle_shortfalls')).toEqual([{ actual_credits: actual, charged_credits: 0, shortfall_credits: actual }]);
    expect(env.DB.q("SELECT settle_locked FROM accounts WHERE id='u1'")[0].settle_locked).toBe(1);
    expect(ledger()).toEqual([{ delta: 0, reason: 'settle_shortfall' }]);
    expect(logged(err, 'settle_fallback_shortfall')).toBe(true);
    // Locked: the next chat is refused before any hold.
    expect((await worker.fetch(chatReq(), env)).status).toBe(403); // settle_locked -> account_frozen
  });
  const bigUsage = { prompt_tokens: 1000, completion_tokens: 200000 };
  const bigFetch = () => vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(JSON.stringify({ choices: [{ message: { content: 'x' } }], usage: bigUsage })));
  it('A1: over-hold fallback charges min(actual, hold); the remainder is a shortfall plus settle_locked', async () => {
    quiet();
    env.DB.q("UPDATE accounts SET balance = ?1 WHERE id='u1'", hold);
    bigFetch(); failSettles(4);
    const r = await worker.fetch(chatReq(), env);
    expect(r.status).toBe(200);
    const want = creditsForSettle('grok-4.3', bigUsage, hold);
    expect(want).toBeGreaterThan(hold);
    expect((await r.json()).rathor.credits_charged).toBe(hold);
    expect(bal()).toBe(0);
    expect(env.DB.q('SELECT charged_credits, shortfall_credits FROM settle_shortfalls')).toEqual([{ charged_credits: hold, shortfall_credits: want - hold }]);
    expect(env.DB.q("SELECT settle_locked FROM accounts WHERE id='u1'")[0].settle_locked).toBe(1);
  });
  it("A1: two open holds: the fallback never consumes the other hold's reserve", async () => {
    quiet();
    const other = 300;
    env.DB.q("UPDATE accounts SET balance = ?1 WHERE id='u1'", hold + other);
    expect(await placeHold(env.DB, 'other-hold', 'u1', other)).toBe(true);
    bigFetch(); failSettles(4);
    expect((await worker.fetch(chatReq(), env)).status).toBe(200);
    expect(bal()).toBe(other);
    expect(env.DB.q("SELECT status FROM holds WHERE id = 'other-hold'")[0].status).toBe('open');
    // The other hold can still settle in full from its own reserve.
    expect(await settleHold(env.DB, 'other-hold', other)).toBe(other);
    expect(bal()).toBe(0);
    expect(env.DB.q('SELECT shortfall_credits FROM settle_shortfalls')[0].shortfall_credits).toBe(creditsForSettle('grok-4.3', bigUsage, hold) - hold);
  });
  it('A2: settle_locked is never set without a shortfall record when the hold is settled by a racer', async () => {
    quiet();
    okFetch();
    const orig = env.DB.batch.bind(env.DB);
    let seen = 0;
    env.DB.batch = async (st) => {
      if (st.some((x) => /'settled'/.test(x.sql))) {
        seen += 1;
        if (seen <= 5) throw new Error('CAS lost');
        // The shortfall batch: a racer settles the hold first.
        env.DB.q("UPDATE holds SET status = 'settled'");
      }
      return orig(st);
    };
    expect((await worker.fetch(chatReq(), env)).status).toBe(200);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM settle_shortfalls')[0].n).toBe(0);
    expect(env.DB.q("SELECT settle_locked FROM accounts WHERE id='u1'")[0].settle_locked).toBe(0);
    expect(ledger()).toEqual([]);
  });
  it('N1: settleHold commits then throws: no double charge, credits_charged is the real charge', async () => {
    quiet();
    okFetch();
    const orig = env.DB.batch.bind(env.DB);
    let once = false;
    env.DB.batch = async (st) => {
      const out = await orig(st);
      if (!once && st.some((x) => /'settled'/.test(x.sql))) { once = true; throw new Error('lost reply after commit'); }
      return out;
    };
    const r = await worker.fetch(chatReq(), env);
    expect(r.status).toBe(200);
    expect((await r.json()).rathor.credits_charged).toBe(actual);
    expect(ledger()).toEqual([{ delta: -actual, reason: 'chat' }]);
    expect(bal()).toBe(1000 - actual);
    expect(env.DB.q("SELECT settle_locked FROM accounts WHERE id='u1'")[0].settle_locked).toBe(0);
  });
});

describe('6) recordBlock row cap and aggregate counter', () => {
  it(`writes at most ${MERCY_BLOCK_ROWS_PER_WINDOW} rows per window; the counter keeps counting`, async () => {
    const { warn } = quiet();
    vi.spyOn(Date, 'now').mockReturnValue(NOW * 1000);
    for (let i = 0; i < MERCY_BLOCK_ROWS_PER_WINDOW + 5; i += 1) await recordBlock(env.DB, { stage: 'request', ruleId: 'r', model: 'grok-4.3' });
    expect(env.DB.q('SELECT COUNT(*) AS n FROM mercy_blocks')[0].n).toBe(MERCY_BLOCK_ROWS_PER_WINDOW);
    expect(env.DB.q("SELECT SUM(count) AS n FROM rate_counters WHERE bucket = 'mercy_blocks'")[0].n).toBe(MERCY_BLOCK_ROWS_PER_WINDOW + 5);
    expect(logged(warn, 'mercy_block_rows_capped')).toBe(true);
  });
  it('request-stage rows keep account_id null and store no content', async () => {
    quiet();
    const secret = 'how to make a pipe bomb';
    const r = await worker.fetch(chatReq({ messages: [{ role: 'user', content: secret }] }), env);
    expect(r.status).toBe(400);
    const rows = env.DB.q('SELECT * FROM mercy_blocks');
    expect(rows.length).toBe(1);
    expect(rows[0].account_id).toBeNull();
    expect(JSON.stringify(rows)).not.toContain('bomb');
    expect(JSON.stringify(env.DB.q('SELECT * FROM rate_counters'))).not.toContain('bomb');
  });
});
