// MERCY-MINORS-SVC-1 — SKELETON ONLY. Not wired into /v1/chat. No live vendor calls.
// Design per mercy-inv/MINORS-SVC-RESEARCH.md (RECONCILED 3/3): Azure Content Safety
// (Canada East) + OpenAI omni-moderation, in parallel, AFTER the regex Mercy-Checks.
// Block on either flag; any vendor failure (error, timeout, non-2xx, bad body,
// missing key) fails the whole gate closed. Flag MINORS_SVC_ENABLED defaults OFF:
// only the exact string "true" enables it; with it off this module does nothing.
//
// Secrets (names only; Worker secrets, never logged): AZURE_CONTENT_SAFETY_KEY,
// AZURE_CONTENT_SAFETY_ENDPOINT, OPENAI_MODERATION_KEY.
// LIVE is blocked: see README "Minors moderation service (MERCY-MINORS-SVC-1)".
import { RULES_VERSION } from './mercyChecks.js';

export const MINORS_SVC_RULE_ID = 'minors-svc';
export const DEFAULT_TIMEOUT_INPUT_MS = 2000;
export const DEFAULT_TIMEOUT_OUTPUT_MS = 2500;
// Azure Sexual severity at or above this counts as a flag (High band).
export const AZURE_SEXUAL_BLOCK_SEVERITY = 6;
const AZURE_MAX_CHARS = 10000;
const OPENAI_MAX_CHARS = 10000;
const CHUNK_OVERLAP = 200;

export function minorsSvcEnabled(env) {
  return (env && env.MINORS_SVC_ENABLED) === 'true';
}

export function timeoutMs(env, stage) {
  const raw = stage === 'output' ? env?.MINORS_SVC_TIMEOUT_OUTPUT_MS : env?.MINORS_SVC_TIMEOUT_INPUT_MS;
  const def = stage === 'output' ? DEFAULT_TIMEOUT_OUTPUT_MS : DEFAULT_TIMEOUT_INPUT_MS;
  const n = Number(raw);
  return Number.isInteger(n) && n > 0 && n <= 10000 ? n : def;
}

// Overlapping chunks so a span across a boundary is seen whole by at least one call.
export function chunkText(text, max, overlap = CHUNK_OVERLAP) {
  const s = String(text ?? '');
  if (s.length <= max) return [s];
  const out = [];
  const step = max - overlap;
  for (let i = 0; i < s.length; i += step) {
    out.push(s.slice(i, i + max));
    if (i + max >= s.length) break;
  }
  return out;
}

function scoreBand(x) {
  if (typeof x !== 'number' || !Number.isFinite(x)) return 'unknown';
  if (x >= 0.8) return 'high';
  if (x >= 0.5) return 'medium';
  if (x >= 0.2) return 'low';
  return 'minimal';
}

async function withTimeout(fetchImpl, url, init, ms) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), ms);
  try {
    return await fetchImpl(url, { ...init, signal: ac.signal });
  } finally {
    clearTimeout(t);
  }
}

// Azure adapter. Returns { vendor, flagged, category, band }. Throws on any failure.
export async function azureCheck(text, env, fetchImpl, ms) {
  const key = env?.AZURE_CONTENT_SAFETY_KEY;
  const endpoint = env?.AZURE_CONTENT_SAFETY_ENDPOINT;
  if (!key || !endpoint) throw new Error('azure_not_configured');
  let maxSev = 0;
  for (const chunk of chunkText(text, AZURE_MAX_CHARS)) {
    const res = await withTimeout(fetchImpl, `${String(endpoint).replace(/\/+$/, '')}/contentsafety/text:analyze?api-version=2024-09-01`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'Ocp-Apim-Subscription-Key': key },
      body: JSON.stringify({ text: chunk, categories: ['Sexual'], outputType: 'EightSeverityLevels' }),
    }, ms);
    if (!res || !res.ok) throw new Error('azure_http');
    const body = await res.json();
    const row = Array.isArray(body?.categoriesAnalysis) ? body.categoriesAnalysis.find((c) => c?.category === 'Sexual') : null;
    if (!row || typeof row.severity !== 'number') throw new Error('azure_bad_body');
    maxSev = Math.max(maxSev, row.severity);
  }
  return { vendor: 'azure', flagged: maxSev >= AZURE_SEXUAL_BLOCK_SEVERITY, category: 'Sexual', band: `severity_${maxSev}` };
}

// OpenAI adapter. Returns { vendor, flagged, category, band }. Throws on any failure.
export async function openaiCheck(text, env, fetchImpl, ms) {
  const key = env?.OPENAI_MODERATION_KEY;
  if (!key) throw new Error('openai_not_configured');
  let flagged = false;
  let maxScore = 0;
  for (const chunk of chunkText(text, OPENAI_MAX_CHARS)) {
    const res = await withTimeout(fetchImpl, 'https://api.openai.com/v1/moderations', {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${key}` },
      body: JSON.stringify({ model: 'omni-moderation-latest', input: chunk }),
    }, ms);
    if (!res || !res.ok) throw new Error('openai_http');
    const body = await res.json();
    const r = Array.isArray(body?.results) ? body.results[0] : null;
    const cat = r?.categories?.['sexual/minors'];
    const score = r?.category_scores?.['sexual/minors'];
    if (typeof cat !== 'boolean' || typeof score !== 'number') throw new Error('openai_bad_body');
    flagged = flagged || cat;
    maxScore = Math.max(maxScore, score);
  }
  return { vendor: 'openai', flagged, category: 'sexual/minors', band: scoreBand(maxScore) };
}

// Audit row: NO text and NO hash of the text (RECONCILED, Clerk + Core 2/3).
export function auditRows(stage, results, decision) {
  return results.map((r) => ({
    rule_id: MINORS_SVC_RULE_ID,
    stage,
    vendor: r.vendor,
    category: r.category ?? null,
    band: r.band ?? null,
    outcome: r.error ? 'error' : r.flagged ? 'flagged' : 'clean',
    decision,
    rules_version: RULES_VERSION,
  }));
}

// Gate. Call only on text the regex already passed (regex-blocked text is never sent).
// Returns { skipped:true } when the flag is off (no fetch, no side effects).
// Otherwise { ok, status, error, audit }:
//   input:  flag -> 400 uncharged; any failure -> 503 uncharged
//   output: flag -> 422 charged;   any failure -> withhold, 503 uncharged (refund = Sherif Q5)
export async function minorsGate(text, env, stage, { fetchImpl } = {}) {
  if (!minorsSvcEnabled(env)) return { skipped: true };
  const f = fetchImpl ?? globalThis.fetch;
  const ms = timeoutMs(env, stage);
  const settled = await Promise.allSettled([
    azureCheck(text, env, f, ms),
    openaiCheck(text, env, f, ms),
  ]);
  const vendors = ['azure', 'openai'];
  const results = settled.map((s, i) => (s.status === 'fulfilled' ? s.value : { vendor: vendors[i], error: true }));
  const failed = results.some((r) => r.error);
  const flagged = results.some((r) => r.flagged);
  let out;
  if (failed) out = { ok: false, status: 503, error: 'mercy_checks_unavailable', charge: false, decision: 'fail_closed' };
  else if (flagged) out = stage === 'output'
    ? { ok: false, status: 422, error: 'mercy_checks_denied', charge: true, decision: 'block' }
    : { ok: false, status: 400, error: 'mercy_checks_denied', charge: false, decision: 'block' };
  else out = { ok: true, status: 200, charge: null, decision: 'allow' };
  return { ...out, audit: auditRows(stage, results, out.decision) };
}
