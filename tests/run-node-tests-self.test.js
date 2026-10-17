/* Self-check for run-node-tests quarantine parsing and unknown-entry detection. */
var fs = require('fs');
var os = require('os');
var path = require('path');
var runner = require('./run-node-tests.js');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

var tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'run-node-tests-self-'));
var quarantineFile = path.join(tmp, 'QUARANTINE.md');
fs.writeFileSync(
  quarantineFile,
  'real.test.js — ok\nghost.test.js — missing file\n',
  'utf8'
);
var quarantine = runner.parseQuarantine(fs.readFileSync(quarantineFile, 'utf8'));
assert(quarantine['real.test.js'] === 'ok', 'parser must read valid entry');
assert(quarantine['ghost.test.js'] === 'missing file', 'parser must read ghost entry');

var unknown = runner.unknownQuarantineEntries(quarantine, ['real.test.js', 'other.test.js']);
assert(unknown.length === 1 && unknown[0] === 'ghost.test.js', 'unknown list: ' + unknown.join(','));

try {
  runner.parseQuarantine('bad-line-no-separator');
  assert(false, 'malformed quarantine must throw');
} catch (e) {
  assert(e.message.indexOf('QUARANTINE.md') !== -1, 'parse error message');
}

fs.rmSync(tmp, { recursive: true, force: true });
console.log('run-node-tests-self ok');
