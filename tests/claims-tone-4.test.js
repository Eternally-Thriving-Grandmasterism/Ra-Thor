'use strict';
const fs = require('fs');
const assert = require('assert');
const { execSync } = require('child_process');
const t = fs.readFileSync('docs/public-launch/README.md', 'utf8');
for (const p of ['mathematically proven', 'self-evolve infinitely', 'valence ≥ 0.999', 'Prosperity is the only trajectory'])
  assert(!t.includes(p), 'phrase still present: ' + p);
assert(t.includes('designed to keep improving'));
try {
  const base = execSync('git merge-base HEAD origin/main', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  const files = execSync(`git diff --name-only ${base} HEAD`, { stdio: ['ignore', 'pipe', 'ignore'] }).toString().split('\n').filter(Boolean);
  const extra = files.filter(f => f !== 'docs/public-launch/README.md' && f !== 'tests/claims-tone-4.test.js');
  assert.deepStrictEqual(extra, [], 'unexpected files changed: ' + extra);
} catch (e) { if (e.name === 'AssertionError') throw e; }
console.log('claims-tone-4 ok');
