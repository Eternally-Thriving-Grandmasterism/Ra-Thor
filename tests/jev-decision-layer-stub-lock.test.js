/* Decision-layer documentation stub lock. Research phrases only. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

var stubPath = path.join(root, 'docs/science/jev_decision_layer_stub.rs');
var notePath = path.join(
  root,
  'docs/science/JEV-DECISION-LAYER-DOCUMENTATION-STUB-2026-10-15.md'
);
var cargoPath = path.join(root, 'Cargo.toml');

assert(fs.existsSync(stubPath), 'local stub must exist');
assert(fs.existsSync(notePath), 'documentation stub must exist');

var stub = fs.readFileSync(stubPath, 'utf8');
var note = fs.readFileSync(notePath, 'utf8');
var cargo = fs.readFileSync(cargoPath, 'utf8');

assert(stub.indexOf('signal only, not a gate') !== -1, 'stub must say signal only, not a gate');
assert(note.indexOf('signal only, not a gate') !== -1, 'note must say signal only, not a gate');

var claim =
  'inspectable research software; optional Grok session; not xAI affiliated; not certified; not a legal product.';
assert(stub.indexOf(claim) !== -1, 'stub must carry the public claim line');
assert(note.indexOf(claim) !== -1, 'note must carry the public claim line');

assert(stub.indexOf('EscalateToLargerModelOrHuman') !== -1, 'low confidence must escalate');
assert(
  stub.indexOf('does not call TypeSafe, OpenRouter, or any external service') !== -1,
  'stub must refuse external decision calls'
);
assert(stub.indexOf('14.15.6') !== -1, 'stub must keep workspace 14.15.6');
assert(stub.indexOf('local_abstain_noul') !== -1, 'pure-local abstain fallback must be named');
assert(
  note.indexOf('PATSAGI-NOTE-2026-10-15-JEV-DECISION-LAYER.md') !== -1,
  'note must record the absent read-first path'
);
assert(note.indexOf('stops at this documentation stub') !== -1, 'note must stop at the stub');

assert(cargo.indexOf('version = "14.15.6"') !== -1, 'workspace version stays 14.15.6');
assert(cargo.indexOf('jev_decision_layer_stub') === -1, 'stub stays out of Cargo.toml');
assert(cargo.indexOf('jev-decision-layer') === -1, 'no decision-layer member path');

console.log('jev decision-layer documentation stub lock passed');
