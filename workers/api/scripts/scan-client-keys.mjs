#!/usr/bin/env node
// WEB-AUTH-1: fail if client-facing site files contain secret key patterns.
// Scoped to root HTML/JS, js/, and common built-output dirs. Does not scan
// tests, docs, workers, crates, or node_modules (those may name prefixes).
import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import { join, relative, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../..', import.meta.url));
const EXTS = new Set(['.html', '.js', '.mjs', '.cjs']);
// Token-start only, so identifiers like store_messages do not match re_.
// re_ / xai- / generic rk_ require a key-length suffix. Prefix names alone in
// docs/tests are out of scope because those trees are not scanned.
const PATTERN = /(^|[^A-Za-z0-9_])(sk_live_|sk_test_|whsec_|rk_(?:test_|live_)|rk_[A-Za-z0-9]{16,}|xai-[A-Za-z0-9]{8,}|re_[A-Za-z0-9]{16,})/;

function walk(dir, out) {
  if (!existsSync(dir)) return;
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === '.git') continue;
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, out);
    else if (EXTS.has(extname(name))) out.push(p);
  }
}

const files = [];
for (const name of readdirSync(ROOT)) {
  if (!EXTS.has(extname(name))) continue;
  files.push(join(ROOT, name));
}
walk(join(ROOT, 'js'), files);
for (const built of ['dist', 'build', 'out']) walk(join(ROOT, built), files);

const hits = [];
for (const file of files) {
  const text = readFileSync(file, 'utf8');
  const lines = text.split(/\r?\n/);
  lines.forEach((line, i) => {
    if (PATTERN.test(line)) hits.push(`${relative(ROOT, file)}:${i + 1}`);
  });
}

if (hits.length) {
  console.error('client-facing files contain secret key patterns:');
  for (const h of hits) console.error(`  ${h}`);
  process.exit(1);
}
console.log(`ok: scanned ${files.length} client-facing files, no key patterns`);
