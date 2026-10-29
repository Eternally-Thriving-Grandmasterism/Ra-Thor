// HOLD-HEADROOM-1 tests. Upstream replies are mocked; no real xAI call is made.
import { describe, it, expect, vi, beforeEach } from 'vitest';
import worker from '../src/index.js';
import { sha256Hex } from '../src/auth.js';
import { creditsForUsage, creditsForTokens, billableOutputTokens, holdEstimate, maxCostEstimate, PER_REQUEST_OVERHEAD_TOKENS, PER_MESSAGE_OVERHEAD_TOKENS, REASONING_HEADROOM_TOKENS } from '../src/credits.js';
import { placeHold, settleHold } from '../src/db.js';
import { fakeD1 } from './fakeD1.js';

const O = 'https://rathor.ai';
const KEY = ['rth', 'test', '0123456789abcdefghijklmnop'].join('_');
let env;
beforeEach(async () => {
  vi.restoreAllMocks();
  env = { ALLOWED_ORIGIN: O, DB: fakeD1(), XAI_API_KEY: 'test', RATES_VERIFIED: 'true' };
  env.DB.q("INSERT INTO accounts (id, balance) VALUES ('u1', 1000)");
  env.DB.q('INSERT INTO api_keys (key_hash, account_id) VALUES (?1, ?2)', await sha256Hex(KEY), 'u1');
});
const req = (body) => new Request('https://api.rathor.ai/v1/chat', { method: 'POST', body: JSON.stringify(body), headers: { Origin: O, Authorization: `Bearer ${KEY}` } });
const chatBody = { model: 'grok-4.7', max_tokens: 100, messages: [{ role: 'user', content: 'hi' }] };
const bal = () => env.DB.q("SELECT balance FROM accounts WHERE id='u1'")[0].balance;
const acct = () => env.DB.q("SELECT frozen, freeze_reason, settle_locked FROM accounts WHERE id='u1'")[0];
const upstream = (usage) => vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(JSON.stringify({ choices: [{ index: 0, message: { role: 'assistant', content: 'hello' }, finish_reason: 'stop' }], usage })));

describe('reasoning tokens in usage', () => {
  it('counts completion_tokens_details.reasoning_tokens as output when reported separately (xAI example shape)', () => {
    const u = { prompt_tokens: 32, completion_tokens: 9, total_tokens: 151, completion_tokens_details: { reasoning_tokens: 110 } };
    expect(billableOutputTokens(u)).toBe(119);
    expect(creditsForUsage('grok-4.7', u)).toBe(creditsForTokens('grok-4.7', 32, 119));
    const big = { prompt_tokens: 1000, completion_tokens: 500, completion_tokens_details: { reasoning_tokens: 20000 } };
    expect(creditsForUsage('grok-4.7', big)).toBe(creditsForTokens('grok-4.7', 1000, 20500));
    expect(creditsForUsage('grok-4.3', big)).toBe(creditsForTokens('grok-4.3', 1000, 20500));
  });
  it('does not double count when total_tokens shows reasoning is inside completion_tokens', () => {
    expect(billableOutputTokens({ prompt_tokens: 10, completion_tokens: 120, total_tokens: 130, completion_tokens_details: { reasoning_tokens: 110 } })).toBe(120);
  });
  it('no reasoning field: unchanged from before', () => {
    expect(creditsForUsage('grok-4.7', { prompt_tokens: 1000, completion_tokens: 500 })).toBe(8);
  });
  it('end to end: reasoning tokens are billed on a chat reply', async () => {
    const usage = { prompt_tokens: 1000, completion_tokens: 500, completion_tokens_details: { reasoning_tokens: 2000 } };
    upstream(usage);
    const j = await (await worker.fetch(req(chatBody), env)).json();
    expect(j.rathor.credits_charged).toBe(creditsForTokens('grok-4.7', 1000, 2500));
    expect(bal()).toBe(1000 - j.rathor.credits_charged);
  });
});

describe('hold covers overhead and reasoning headroom', () => {
  it('adds named per-request and per-message overhead and reasoning headroom', () => {
    expect(PER_REQUEST_OVERHEAD_TOKENS).toBeGreaterThan(0);
    expect(PER_MESSAGE_OVERHEAD_TOKENS).toBeGreaterThan(0);
    expect(REASONING_HEADROOM_TOKENS).toBeGreaterThan(0);
    for (const m of ['grok-4.7', 'grok-4.3']) {
      expect(holdEstimate(m, 50, 3, 4096)).toBe(creditsForTokens(m, 50 + PER_REQUEST_OVERHEAD_TOKENS + 3 * PER_MESSAGE_OVERHEAD_TOKENS, 4096 + REASONING_HEADROOM_TOKENS));
      expect(holdEstimate(m, 50, 3, 4096)).toBeGreaterThan(maxCostEstimate(m, 50, 4096));
      expect(holdEstimate(m, 50, 4, 4096)).toBeGreaterThanOrEqual(holdEstimate(m, 50, 3, 4096));
    }
  });
  it('a reply using the template overhead plus max_tokens plus reasoning up to the headroom settles within the hold', async () => {
    const bytes = new TextEncoder().encode(JSON.stringify(chatBody.messages)).length;
    const usage = { prompt_tokens: bytes + PER_REQUEST_OVERHEAD_TOKENS + PER_MESSAGE_OVERHEAD_TOKENS, completion_tokens: 100, completion_tokens_details: { reasoning_tokens: REASONING_HEADROOM_TOKENS } };
    upstream(usage);
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const j = await (await worker.fetch(req(chatBody), env)).json();
    expect(j.rathor.credits_held).toBe(holdEstimate('grok-4.7', bytes, 1, 100));
    expect(j.rathor.credits_charged).toBe(creditsForUsage('grok-4.7', usage));
    expect(j.rathor.credits_charged).toBeLessThanOrEqual(j.rathor.credits_held);
    expect(err.mock.calls.filter((c) => String(c[0]).includes('settle_over_hold'))).toHaveLength(0);
  });
});

describe('settle above the hold', () => {
  const overUsage = { prompt_tokens: 1000, completion_tokens: 100, completion_tokens_details: { reasoning_tokens: 60000 } };
  it('enough balance: charged at actual cost, overage debited, not frozen', async () => {
    upstream(overUsage);
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const j = await (await worker.fetch(req(chatBody), env)).json();
    const cost = creditsForUsage('grok-4.7', overUsage);
    expect(cost).toBeGreaterThan(j.rathor.credits_held);
    expect(j.rathor.credits_charged).toBe(cost);
    expect(bal()).toBe(1000 - cost);
    expect(env.DB.q("SELECT delta FROM ledger WHERE reason='chat'")).toEqual([{ delta: -cost }]);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM settle_shortfalls')[0].n).toBe(0);
    expect(acct().frozen).toBe(0); expect(acct().settle_locked).toBe(0);
    const line = err.mock.calls.map((c) => String(c[0])).find((l) => l.includes('settle_over_hold'));
    expect(JSON.parse(line)).toMatchObject({ msg: 'settle_over_hold', alert: false, shortfall: 0, charged: cost });
    expect(env.DB.q("SELECT COUNT(*) AS n FROM holds WHERE status='open'")[0].n).toBe(0);
  });
  it('not enough balance: shortfall recorded, account locked, alert logged, next request refused', async () => {
    env.DB.q("UPDATE accounts SET balance = 300 WHERE id='u1'");
    upstream(overUsage);
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const j = await (await worker.fetch(req(chatBody), env)).json();
    const cost = creditsForUsage('grok-4.7', overUsage);
    expect(cost).toBeGreaterThan(300);
    expect(j.rathor.credits_charged).toBe(300);
    expect(bal()).toBe(0);
    const [row] = env.DB.q('SELECT * FROM settle_shortfalls');
    expect(row).toMatchObject({ account_id: 'u1', actual_credits: cost, charged_credits: 300, shortfall_credits: cost - 300, model: 'grok-4.7', resolved: 0 });
    expect(env.DB.q("SELECT delta FROM ledger WHERE reason='settle_shortfall'")).toEqual([{ delta: 0 }]);
    expect(acct()).toEqual({ frozen: 0, freeze_reason: null, settle_locked: 1 });
    const line = err.mock.calls.map((c) => String(c[0])).find((l) => l.includes('settle_over_hold'));
    expect(JSON.parse(line)).toMatchObject({ msg: 'settle_over_hold', alert: true, shortfall: cost - 300, frozen: true });
    expect(line).not.toMatch(/hello|Bearer|rth_/);
    // Top up: still locked until a human resolves it.
    env.DB.q("UPDATE accounts SET balance = 5000 WHERE id='u1'");
    const f = upstream({ prompt_tokens: 1, completion_tokens: 1 });
    const r2 = await worker.fetch(req(chatBody), env);
    expect(r2.status).toBe(403);
    expect((await r2.json()).error).toBe('account_frozen');
    expect(f).not.toHaveBeenCalled();
    expect(await placeHold(env.DB, 'h2', 'u1', 1)).toBe(false);
  });
  it('overage never eats into another open hold', async () => {
    env.DB.q("UPDATE accounts SET balance = 100 WHERE id='u1'");
    expect(await placeHold(env.DB, 'a', 'u1', 10)).toBe(true);
    expect(await placeHold(env.DB, 'b', 'u1', 60)).toBe(true);
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const c = await settleHold(env.DB, 'a', 500, { model: 'grok-4.3' });
    expect(c).toBe(40); // own hold 10 + unreserved 30
    expect(bal()).toBe(60);
    expect(await settleHold(env.DB, 'b', 60)).toBe(60);
    expect(bal()).toBe(0);
    expect(env.DB.q('SELECT shortfall_credits FROM settle_shortfalls')[0].shortfall_credits).toBe(460);
  });
  it('a second settle of the same hold is a no-op', async () => {
    expect(await placeHold(env.DB, 'a', 'u1', 10)).toBe(true);
    vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(await settleHold(env.DB, 'a', 50)).toBe(50);
    expect(await settleHold(env.DB, 'a', 50)).toBe(null);
    expect(bal()).toBe(950);
  });
  it('missing usage still charges exactly the full hold (#757 fallback)', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(JSON.stringify({ choices: [{ index: 0, message: { role: 'assistant', content: 'hello' } }] })));
    const j = await (await worker.fetch(req(chatBody), env)).json();
    expect(j.rathor.credits_charged).toBe(j.rathor.credits_held);
  });
  it('malformed reasoning_tokens is treated as missing usage (full hold)', async () => {
    upstream({ prompt_tokens: 1, completion_tokens: 1, completion_tokens_details: { reasoning_tokens: -5 } });
    const j = await (await worker.fetch(req(chatBody), env)).json();
    expect(j.rathor.credits_charged).toBe(j.rathor.credits_held);
  });
});

describe('RATES_VERIFIED gate', () => {
  it('false: /v1/chat is 503 before any hold, settle, overage or freeze', async () => {
    env.RATES_VERIFIED = 'false';
    const f = upstream({ prompt_tokens: 1000, completion_tokens: 100, completion_tokens_details: { reasoning_tokens: 60000 } });
    const r = await worker.fetch(req(chatBody), env);
    expect(r.status).toBe(503);
    expect((await r.json()).error).toBe('billing_disabled_rates_unverified');
    expect(f).not.toHaveBeenCalled();
    expect(bal()).toBe(1000);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM holds')[0].n).toBe(0);
    expect(env.DB.q('SELECT COUNT(*) AS n FROM settle_shortfalls')[0].n).toBe(0);
    expect(acct().frozen).toBe(0); expect(acct().settle_locked).toBe(0);
  });
});
