#!/usr/bin/env node
// Explicit baseline regenerator for Mercy-Checks eval (MERCY-MOD-1).
// NEVER invoked by CI or vitest. Usage (from workers/api):
//   node scripts/regenerate-mercy-baseline.mjs
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { scanText, mercyChecks, RULES_VERSION } from '../src/mercyChecks.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const setPath = join(root, 'test/mercy-eval/set.jsonl');
const outDir = join(root, 'test/mercy-eval/baselines');
const rows = readFileSync(setPath, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));

function scoreLine(row) {
  if (row.side === 'output') {
    const r = mercyChecks.checkResponse({ choices: [{ message: { content: row.text } }] });
    return r.allowed ? null : r.reason;
  }
  return scanText(row.text);
}

const byRule = {};
for (const row of rows) {
  if (!byRule[row.rule_id]) byRule[row.rule_id] = { benign: 0, harmful: 0, falseBlocks: 0, misses: 0 };
  const b = byRule[row.rule_id];
  const got = scoreLine(row);
  if (row.label === 'benign') {
    b.benign += 1;
    if (got) b.falseBlocks += 1;
  } else {
    b.harmful += 1;
    if (!got || got !== row.rule_id) b.misses += 1;
  }
}

const reason = process.argv.slice(2).join(' ').trim()
  || `Baseline for ${RULES_VERSION}. Update only with a seat-voted reason.`;

const compact = {
  rules_version: RULES_VERSION,
  reason,
  by_rule_id: Object.fromEntries(Object.entries(byRule).map(([k, v]) => [k, {
    set_benign: v.benign,
    set_harmful: v.harmful,
    false_block_rate: v.benign ? v.falseBlocks / v.benign : 0,
    miss_rate: v.harmful ? v.misses / v.harmful : 0,
  }])),
};

mkdirSync(outDir, { recursive: true });
const out = join(outDir, `${RULES_VERSION}.json`);
writeFileSync(out, JSON.stringify(compact, null, 2) + '\n');
console.log('wrote', out);
console.log(JSON.stringify(compact, null, 2));
