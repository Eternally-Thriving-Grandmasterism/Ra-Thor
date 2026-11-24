'use strict';
const fs = require('fs');
const assert = require('assert');
const t = fs.readFileSync('docs/public-launch/README.md', 'utf8');
for (const p of ['mathematically proven', 'self-evolve infinitely', 'valence ≥ 0.999', 'Prosperity is the only trajectory'])
  assert(!t.includes(p), 'phrase still present: ' + p);
assert(t.includes('designed to keep improving'));
console.log('claims-tone-4 ok');
const row1 = t.split('\n').find(l => l.includes('designed to keep improving'));
assert(row1.includes('guided by the TOLC 8 Mercy-Checks'), 'row 1 wording');
assert(!/mercy gates/i.test(row1), 'row 1 must not say mercy gates');
