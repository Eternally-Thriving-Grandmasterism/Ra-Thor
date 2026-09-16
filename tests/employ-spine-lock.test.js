/* EMPLOY-1: stranger employ spine. One whitepaper; living pages follow it. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
var employMd = fs.readFileSync(path.join(root, 'docs/EMPLOY.md'), 'utf8');
var employHtml = fs.readFileSync(path.join(root, 'employ.html'), 'utf8');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

assert(employMd.indexOf('inspect ≠ METR') !== -1, 'docs/EMPLOY.md must contain inspect ≠ METR');
assert(employMd.indexOf('info@Rathor.ai') !== -1, 'docs/EMPLOY.md must contain info@Rathor.ai');
assert(employMd.indexOf('14.15.6') !== -1, 'docs/EMPLOY.md must contain 14.15.6');
assert(employMd.indexOf('Layer 0') !== -1, 'docs/EMPLOY.md must contain Layer 0');
assert(employMd.indexOf('cargo test --workspace') !== -1, 'docs/EMPLOY.md must name cargo test --workspace');
assert(employMd.indexOf('Powrush-MMO is a separate repo') !== -1, 'docs/EMPLOY.md must keep dual-repo');
assert(employMd.indexOf('independent of xAI') !== -1, 'docs/EMPLOY.md must say independent of xAI');

assert(employHtml.indexOf('article') !== -1 && /<article\b[^>]*\brt-prose\b/.test(employHtml),
  'employ.html must keep article.rt-prose');
assert(employHtml.indexOf('mailto:info@Rathor.ai') !== -1, 'employ.html must keep mailto:info@Rathor.ai');

assert(employMd.indexOf('## The loop') !== -1, 'docs/EMPLOY.md must restore The loop');
assert(employMd.indexOf('Intend') !== -1, 'docs/EMPLOY.md loop must name Intend');
assert(employHtml.indexOf('Pass the gates') !== -1, 'employ.html must restore Pass the gates');
assert(employHtml.indexOf('14.18') === -1, 'employ.html must not sell 14.18');

console.log('EMPLOY-1 employ-spine-lock checks passed');
