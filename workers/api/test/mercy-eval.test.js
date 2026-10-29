// MERCY-MOD-1 R2: deterministic offline scorer. No network. Never writes baselines.
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { scanText, mercyChecks, RULES_VERSION } from '../src/mercyChecks.js';
import { CATEGORY } from '../src/mercyRules.js';

const dir = dirname(fileURLToPath(import.meta.url));
const evalDir = join(dir, 'mercy-eval');
const setPath = join(evalDir, 'set.jsonl');
const holdoutPath = join(evalDir, 'holdout.jsonl');
const baseDir = join(evalDir, 'baselines');
const HARD_CATEGORIES = new Set(['cbrn', 'explosives', 'minors-hard']);

function loadJsonl(path) {
  return readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

function scoreLine(row) {
  if (row.side === 'output') {
    const r = mercyChecks.checkResponse({ choices: [{ message: { content: row.text } }] });
    return r.allowed ? null : r.reason;
  }
  return scanText(row.text);
}

function scoreAll(rows) {
  const byRule = {};
  const byCat = {};
  const ensure = (bag, key) => {
    if (!bag[key]) bag[key] = { benign: 0, harmful: 0, falseBlocks: 0, misses: 0, ids_miss: [], ids_fb: [] };
    return bag[key];
  };
  for (const row of rows) {
    const got = scoreLine(row);
    const R = ensure(byRule, row.rule_id);
    const C = ensure(byCat, row.category || 'uncategorised');
    if (row.label === 'benign') {
      R.benign += 1; C.benign += 1;
      if (got) { R.falseBlocks += 1; C.falseBlocks += 1; R.ids_fb.push(row.id); C.ids_fb.push(row.id); }
    } else {
      R.harmful += 1; C.harmful += 1;
      if (!got || got !== row.rule_id) {
        R.misses += 1; C.misses += 1; R.ids_miss.push(row.id); C.ids_miss.push(row.id);
      }
    }
  }
  const rates = (bag) => Object.fromEntries(Object.entries(bag).map(([k, v]) => [k, {
    set_benign: v.benign,
    set_harmful: v.harmful,
    false_block_rate: v.benign ? v.falseBlocks / v.benign : 0,
    miss_rate: v.harmful ? v.misses / v.harmful : 0,
    false_blocks: v.falseBlocks,
    misses: v.misses,
    ids_miss: v.ids_miss,
    ids_fb: v.ids_fb,
  }]));
  return { rules_version: RULES_VERSION, by_rule_id: rates(byRule), by_category: rates(byCat) };
}

function printReport(label, report) {
  // eslint-disable-next-line no-console
  console.log(JSON.stringify({
    set: label,
    rules_version: report.rules_version,
    by_rule_id: Object.fromEntries(Object.entries(report.by_rule_id).map(([k, v]) => [k, {
      set_benign: v.set_benign, set_harmful: v.set_harmful,
      false_block_rate: Number(v.false_block_rate.toFixed(4)),
      miss_rate: Number(v.miss_rate.toFixed(4)),
      false_blocks: v.false_blocks, misses: v.misses,
    }])),
    by_category: Object.fromEntries(Object.entries(report.by_category).map(([k, v]) => [k, {
      set_benign: v.set_benign, set_harmful: v.set_harmful,
      false_block_rate: Number(v.false_block_rate.toFixed(4)),
      miss_rate: Number(v.miss_rate.toFixed(4)),
      false_blocks: v.false_blocks, misses: v.misses,
      ids_miss: v.ids_miss, ids_fb: v.ids_fb,
    }])),
  }, null, 2));
}

describe('MERCY-MOD-1 R1 eval set shape', () => {
  const rows = loadJsonl(setPath);
  it('has at least 50 benign and 50 harmful per rule_id', () => {
    const c = {};
    for (const r of rows) {
      const k = `${r.rule_id}|${r.label}`;
      c[k] = (c[k] || 0) + 1;
    }
    for (const id of Object.values(CATEGORY)) {
      expect(c[`${id}|benign`], `${id} benign`).toBeGreaterThanOrEqual(50);
      expect(c[`${id}|harmful`], `${id} harmful`).toBeGreaterThanOrEqual(50);
    }
  });
});

describe('MERCY-MOD-1 R2 training harness', () => {
  const rows = loadJsonl(setPath);
  const report = scoreAll(rows);

  it('prints per-category training rates with set sizes', () => {
    printReport('training', report);
    expect(report.rules_version).toBe(RULES_VERSION);
  });

  it('hard categories have 0 misses on the training set', () => {
    for (const [cat, v] of Object.entries(report.by_category)) {
      if (!HARD_CATEGORIES.has(cat)) continue;
      expect(v.misses, `training misses in ${cat}: ${v.ids_miss.join(',')}`).toBe(0);
    }
  });

  it('requires a committed baseline for RULES_VERSION (never auto-writes)', () => {
    const path = join(baseDir, `${RULES_VERSION}.json`);
    expect(existsSync(path), `missing baseline ${path}; run: node scripts/regenerate-mercy-baseline.mjs`).toBe(true);
    const base = JSON.parse(readFileSync(path, 'utf8'));
    expect(base.rules_version).toBe(RULES_VERSION);
    expect(String(base.reason || '').length).toBeGreaterThan(0);
    for (const [rid, cur] of Object.entries(report.by_rule_id)) {
      const b = base.by_rule_id[rid];
      expect(b, `missing baseline for ${rid}`).toBeTruthy();
      const worseMiss = cur.miss_rate > b.miss_rate + 1e-12;
      const worseFb = cur.false_block_rate > b.false_block_rate + 1e-12;
      if (worseMiss || worseFb) {
        expect.fail(
          `${rid}: miss ${cur.miss_rate} vs base ${b.miss_rate}, fb ${cur.false_block_rate} vs base ${b.false_block_rate}. `
          + 'Regenerate with node scripts/regenerate-mercy-baseline.mjs "<reason>" after a seat vote.',
        );
      }
    }
  });
});

describe('MERCY-MOD-1 holdout (never tune on this set)', () => {
  const rows = loadJsonl(holdoutPath);
  const report = scoreAll(rows);

  it('prints per-category holdout rates separately from training', () => {
    printReport('holdout', report);
    expect(rows.length).toBeGreaterThan(10);
  });

  it('hard categories have 0 misses on the holdout', () => {
    for (const [cat, v] of Object.entries(report.by_category)) {
      if (!HARD_CATEGORIES.has(cat)) continue;
      expect(v.misses, `holdout misses in ${cat}: ${v.ids_miss.join(',')}`).toBe(0);
    }
  });

  it('reports holdout false blocks honestly (no removal to pretty-up rates)', () => {
    const fbs = Object.values(report.by_category).flatMap((v) => v.ids_fb);
    if (fbs.length) {
      // eslint-disable-next-line no-console
      console.log('holdout false blocks:', fbs);
    }
    expect(Array.isArray(fbs)).toBe(true);
  });
});
