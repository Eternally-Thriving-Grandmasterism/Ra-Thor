// MERCY-MINORS-SVC-1: mocked vendor replies only. No keys, no network.
import { describe, it, expect, vi } from 'vitest';
import { minorsGate, minorsSvcEnabled, chunkText, timeoutMs, auditRows } from '../src/minorsService.js';
import { RULES_VERSION } from '../src/mercyChecks.js';
import fs from 'node:fs';

const ENV = { MINORS_SVC_ENABLED: 'true', AZURE_CONTENT_SAFETY_KEY: 'placeholder', AZURE_CONTENT_SAFETY_ENDPOINT: 'https://azure.invalid', OPENAI_MODERATION_KEY: 'placeholder' };
const ok = (body) => ({ ok: true, status: 200, json: async () => body });
const az = (sev) => ok({ categoriesAnalysis: [{ category: 'Sexual', severity: sev }] });
const oa = (flag, score) => ok({ results: [{ categories: { 'sexual/minors': flag }, category_scores: { 'sexual/minors': score } }] });
function mock({ azure, openai }) {
  return vi.fn(async (url) => (String(url).includes('azure.invalid') ? azure() : openai()));
}
const TEXT = 'sample text A';

describe('flag', () => {
  it('defaults off; only exact "true" enables', () => {
    for (const v of [undefined, '', 'false', 'TRUE', '1', 'yes', ' true']) expect(minorsSvcEnabled({ MINORS_SVC_ENABLED: v })).toBe(false);
    expect(minorsSvcEnabled({ MINORS_SVC_ENABLED: 'true' })).toBe(true);
    expect(minorsSvcEnabled(undefined)).toBe(false);
  });
  it('off => skipped, no fetch', async () => {
    const f = vi.fn();
    expect(await minorsGate(TEXT, { ...ENV, MINORS_SVC_ENABLED: 'false' }, 'input', { fetchImpl: f })).toEqual({ skipped: true });
    expect(f).not.toHaveBeenCalled();
  });
  it('wrangler.toml sets it "false" in every env', () => {
    const t = fs.readFileSync(new URL('../wrangler.toml', import.meta.url), 'utf8');
    const m = t.match(/^MINORS_SVC_ENABLED\s*=\s*"([^"]*)"/gm);
    expect(m).toHaveLength(2);
    for (const l of m) expect(l).toMatch(/"false"$/);
  });
  it('not imported by index.js (skeleton, request path unchanged)', () => {
    const src = fs.readFileSync(new URL('../src/index.js', import.meta.url), 'utf8');
    expect(src).not.toMatch(/minorsService/);
  });
});

describe('decisions (mocked)', () => {
  it('both clean => allow', async () => {
    const r = await minorsGate(TEXT, ENV, 'input', { fetchImpl: mock({ azure: () => az(0), openai: () => oa(false, 0.01) }) });
    expect(r).toMatchObject({ ok: true, decision: 'allow' });
  });
  it('openai flag on input => 400 uncharged', async () => {
    const r = await minorsGate(TEXT, ENV, 'input', { fetchImpl: mock({ azure: () => az(0), openai: () => oa(true, 0.9) }) });
    expect(r).toMatchObject({ ok: false, status: 400, charge: false, decision: 'block' });
  });
  it('azure high severity on output => 422 charged', async () => {
    const r = await minorsGate(TEXT, ENV, 'output', { fetchImpl: mock({ azure: () => az(6), openai: () => oa(false, 0.01) }) });
    expect(r).toMatchObject({ ok: false, status: 422, charge: true });
  });
  it('azure severity 4 is not a flag', async () => {
    const r = await minorsGate(TEXT, ENV, 'input', { fetchImpl: mock({ azure: () => az(4), openai: () => oa(false, 0.01) }) });
    expect(r.ok).toBe(true);
  });
  for (const stage of ['input', 'output']) {
    it(`${stage}: one error + one clean => 503 uncharged`, async () => {
      const r = await minorsGate(TEXT, ENV, stage, { fetchImpl: mock({ azure: () => { throw new Error('x'); }, openai: () => oa(false, 0) }) });
      expect(r).toMatchObject({ ok: false, status: 503, charge: false, decision: 'fail_closed' });
    });
    it(`${stage}: non-2xx => 503`, async () => {
      const r = await minorsGate(TEXT, ENV, stage, { fetchImpl: mock({ azure: () => az(0), openai: () => ({ ok: false, status: 500, json: async () => ({}) }) }) });
      expect(r.status).toBe(503);
    });
    it(`${stage}: malformed body => 503`, async () => {
      const r = await minorsGate(TEXT, ENV, stage, { fetchImpl: mock({ azure: () => ok({}), openai: () => oa(false, 0) }) });
      expect(r.status).toBe(503);
    });
  }
  it('missing key => 503, no fetch for that vendor', async () => {
    const f = mock({ azure: () => az(0), openai: () => oa(false, 0) });
    const r = await minorsGate(TEXT, { ...ENV, OPENAI_MODERATION_KEY: '' }, 'input', { fetchImpl: f });
    expect(r.status).toBe(503);
    expect(f).toHaveBeenCalledTimes(1);
  });
  it('timeout aborts and fails closed', async () => {
    const hang = (url, init) => new Promise((_, rej) => init.signal.addEventListener('abort', () => rej(new Error('aborted'))));
    const t0 = Date.now();
    const r = await minorsGate(TEXT, { ...ENV, MINORS_SVC_TIMEOUT_INPUT_MS: '50' }, 'input', { fetchImpl: hang });
    expect(r.status).toBe(503);
    expect(Date.now() - t0).toBeLessThan(1000);
  });
});

describe('helpers', () => {
  it('timeouts default 2000/2500 and are env-tunable', () => {
    expect(timeoutMs({}, 'input')).toBe(2000);
    expect(timeoutMs({}, 'output')).toBe(2500);
    expect(timeoutMs({ MINORS_SVC_TIMEOUT_OUTPUT_MS: '900' }, 'output')).toBe(900);
    expect(timeoutMs({ MINORS_SVC_TIMEOUT_INPUT_MS: 'abc' }, 'input')).toBe(2000);
  });
  it('chunks overlap and cover the text', () => {
    const s = 'a'.repeat(25000);
    const c = chunkText(s, 10000, 200);
    expect(c.length).toBe(3);
    expect(c.every((x) => x.length <= 10000)).toBe(true);
    expect(chunkText('sample text B', 10000)).toEqual(['sample text B']);
  });
  it('audit rows carry no text and no hash', async () => {
    const secret = 'sample text C unique-marker';
    const r = await minorsGate(secret, ENV, 'input', { fetchImpl: mock({ azure: () => az(2), openai: () => oa(false, 0.3) }) });
    expect(r.audit).toHaveLength(2);
    for (const row of r.audit) {
      expect(Object.keys(row).sort()).toEqual(['band', 'category', 'decision', 'outcome', 'rule_id', 'rules_version', 'stage', 'vendor']);
      expect(row.rules_version).toBe(RULES_VERSION);
      expect(JSON.stringify(row)).not.toContain('unique-marker');
      expect(JSON.stringify(row)).not.toMatch(/[0-9a-f]{32,}/);
    }
    expect(auditRows('input', [{ vendor: 'azure', error: true }], 'fail_closed')[0].outcome).toBe('error');
  });
});
