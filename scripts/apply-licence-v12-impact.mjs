#!/usr/bin/env node
/**
 * Apply AG-SML v1.2 bulk text updates from impact list (excluding PUBLIC_CLAIM.lock.md).
 */
import fs from 'fs';
import path from 'path';

const root = path.join(import.meta.dirname, '..');
const impactPath = process.env.IMPACT || '/home/ubuntu/.cursor/projects/workspace/uploads/AG-SML-v1.2-impact_23ab.md';

const SKIP = new Set([
  'PUBLIC_CLAIM.lock.md',
  'licence.html', // i18n-driven defaults
  'i18n/en.json',
]);

const SKIP_PREFIX = ['docs/archive/'];

function filesFromImpact(text) {
  const out = new Set();
  for (const line of text.split('\n')) {
    const m = line.match(/^### `([^`]+)`/);
    if (!m) continue;
    let f = m[1].replace(/ — \d+ hit\(s\)$/, '');
    if (SKIP.has(f)) continue;
    if (SKIP_PREFIX.some((p) => f.startsWith(p))) continue;
    if (f.startsWith('tests/') && f !== 'tests/license-form-lock.test.js') continue;
    if (f.startsWith('i18n/')) continue;
    out.add(f);
  }
  return [...out].sort();
}

const PHRASES = [
  [/AG-SML v1\.1(?!\s+house text, scoped)/g, 'AG-SML v1.2'],
  [
    /Personal \/ education \/ modest freelance remains free under AG-SML v1\.2\./g,
    'Non-revenue personal / education / research use may stay free under AG-SML v1.2 §1.',
  ],
  [
    /daily living or independent creative work \(freelancer, artist, writer, or similar\), even where modest personal income depends on the tool/g,
    'non-commercial creative work',
  ],
  [
    /daily living or independent creative work, even where modest personal income depends on the tool/g,
    'non-commercial creative work',
  ],
  [
    /daily living or independent creative work by a natural person stays free under the public \*\*AG-SML v1\.2\*\* grant when it fits the licence text in the monorepo — even where modest personal income depends on the tool\./g,
    'non-commercial creative work by a natural person may stay under the public **AG-SML v1.2** grant when it fits the licence text in the monorepo and earns no revenue (see LICENSE §1A).',
  ],
  [
    /personal, educational, research, or modest independent-professional work/g,
    'personal, educational, research, or non-commercial creative work',
  ],
  [
    /Modest personal income from freelance, artistic, or independent creative work stays on the free §1 grant for natural persons\./g,
    'Revenue or income from using, running, or outputs of the Software requires a paid commercial licence (see LICENSE §1A). Non-commercial personal use stays on the free §1 grant for natural persons.',
  ],
  [
    /"Modest" is intentionally undefined\./g,
    'Revenue edge cases: email info@Rathor.ai.',
  ],
  [
    /modest independent-professional aid without payment/g,
    'non-commercial creative work without payment',
  ],
  [
    /Do not mint AG-SML v1\.2 for wording cleanup\. v1\.2 is reserved for an objective RBE oracle if one is later operated\./g,
    'AG-SML v1.2 is the living public grant; transition rules are in root LICENSE §7.',
  ],
  [
    /Do not invent AG-SML v1\.2 for wording cleanup\. v1\.2 is reserved for an \*objective\* oracle if one is later operated\./g,
    'AG-SML v1.2 is the living public grant; transition rules are in root LICENSE §7.',
  ],
  [
    /- \*\*Do not invent AG-SML v1\.2\*\* for book wording \(reserved per `COMMERCIAL_LICENSE\.md` §8\)\./g,
    '- **AG-SML v1.2** is the living public grant (see root `LICENSE` §7).',
  ],
  [
    /license = "AG-SML v1\.1 — Autonomicity Games Sovereign Mercy License"/g,
    'license = "AG-SML v1.2 — Autonomicity Games Sovereign Mercy License"',
  ],
  [
    /# AG-SML v1\.1 \|/g,
    '# AG-SML v1.2 |',
  ],
  [
    /License: AG-SML v1\.1 \(personal \/ research\)/g,
    'License: AG-SML v1.2 (non-revenue personal / research)',
  ],
];

function applyPhrases(text) {
  let t = text;
  for (const [re, rep] of PHRASES) {
    t = t.replace(re, rep);
  }
  return t;
}

const impact = fs.readFileSync(impactPath, 'utf8');
const files = filesFromImpact(impact);
let changed = 0;
for (const rel of files) {
  const abs = path.join(root, rel);
  if (!fs.existsSync(abs)) continue;
  const before = fs.readFileSync(abs, 'utf8');
  const after = applyPhrases(before);
  if (after !== before) {
    fs.writeFileSync(abs, after);
    changed++;
  }
}
console.log('impact files touched:', changed, 'of', files.length);
