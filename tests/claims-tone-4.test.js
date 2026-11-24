'use strict';
const fs = require('fs');
const assert = require('assert');
const t = fs.readFileSync('docs/public-launch/README.md', 'utf8');
for (const p of ['mathematically proven', 'self-evolve infinitely', 'valence ≥ 0.999', 'Prosperity is the only trajectory'])
  assert(!t.includes(p), 'phrase still present: ' + p);
assert(t.includes('designed to keep improving'));
console.log('claims-tone-4 ok');
