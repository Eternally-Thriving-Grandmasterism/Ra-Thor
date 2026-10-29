/* BRIEF-1: agent run brief lock. Inner-loop contract must stay fill-in, not essay. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
var briefPath = path.join(root, 'docs/AGENT_RUN_BRIEF.md');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

assert(fs.existsSync(briefPath), 'docs/AGENT_RUN_BRIEF.md must exist');
var brief = fs.readFileSync(briefPath, 'utf8');

assert(brief.indexOf('FINISH LINE') !== -1, 'must contain FINISH LINE');
assert(brief.indexOf('HOLD') !== -1, 'must contain HOLD');
assert(brief.indexOf('STOP') !== -1, 'must contain STOP');
assert(brief.indexOf('inspect ≠ METR') !== -1, 'must contain inspect ≠ METR');
assert(brief.indexOf('info@Rathor.ai') !== -1, 'must contain info@Rathor.ai');
assert(brief.indexOf('14.15.6') !== -1, 'must contain 14.15.6');
assert(brief.indexOf('outer loop does not write code') !== -1, 'must name outer-loop bound');
assert(brief.indexOf('Powrush-MMO is a separate repo') !== -1, 'must keep dual-repo');

console.log('BRIEF-1 agent-brief-lock checks passed');
