/* Self-check for run-node-tests quarantine parsing, unknown-entry detection, and runner integration. */
var fs = require('fs');
var os = require('os');
var path = require('path');
var cp = require('child_process');
var runner = require('./run-node-tests.js');

var realRunner = path.join(__dirname, 'run-node-tests.js');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function writePassTest(dir) {
  fs.writeFileSync(path.join(dir, 'pass.test.js'), 'console.log("pass");\n', 'utf8');
}

function spawnRunner(testsDir) {
  return cp.spawnSync(process.execPath, [realRunner], {
    env: Object.assign({}, process.env, { RUN_NODE_TESTS_DIR: testsDir }),
    encoding: 'utf8',
  });
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

var cleanDir = fs.mkdtempSync(path.join(os.tmpdir(), 'run-node-tests-clean-'));
writePassTest(cleanDir);
var cleanRun = spawnRunner(cleanDir);
assert(cleanRun.status === 0, 'clean fixture must exit 0, got ' + cleanRun.status + ' ' + cleanRun.stderr);
fs.rmSync(cleanDir, { recursive: true, force: true });

var ghostDir = fs.mkdtempSync(path.join(os.tmpdir(), 'run-node-tests-ghost-'));
writePassTest(ghostDir);
fs.writeFileSync(path.join(ghostDir, 'QUARANTINE.md'), 'ghost.test.js — no such file\n', 'utf8');
var ghostRun = spawnRunner(ghostDir);
assert(ghostRun.status === 1, 'ghost fixture must exit 1, got ' + ghostRun.status);
var ghostOut = (ghostRun.stdout || '') + (ghostRun.stderr || '');
assert(ghostOut.indexOf('UNKNOWN QUARANTINE ENTRY: ghost.test.js') !== -1, 'ghost fixture must log unknown entry');
fs.rmSync(ghostDir, { recursive: true, force: true });

var caseDir = fs.mkdtempSync(path.join(os.tmpdir(), 'run-node-tests-case-'));
writePassTest(caseDir);
fs.writeFileSync(path.join(caseDir, 'QUARANTINE.md'), 'Pass.test.js — wrong case\n', 'utf8');
var caseRun = spawnRunner(caseDir);
assert(caseRun.status === 1, 'wrong-case fixture must exit 1, got ' + caseRun.status);
var caseOut = (caseRun.stdout || '') + (caseRun.stderr || '');
assert(caseOut.indexOf('UNKNOWN QUARANTINE ENTRY: Pass.test.js') !== -1, 'wrong-case fixture must log unknown entry');
fs.rmSync(caseDir, { recursive: true, force: true });

console.log('run-node-tests-self ok');
