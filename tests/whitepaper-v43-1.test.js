/* WHITEPAPER-V43-1: plain-language whitepaper v4.3 is the current paper.
 * Contact: info@Rathor.ai */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
function assert(cond, msg) { if (!cond) throw new Error(msg); }
function read(rel) { return fs.readFileSync(path.join(root, rel), 'utf8'); }

var md = read('docs/WHITEPAPER_v4.3.md');
var html = read('docs/WHITEPAPER_v4.3.html');

/* Sections in the ruled order, no meta-talk up front. */
var order = ['## What Ra-Thor is', '## What works today', '## How to try it', "## How it's built", '## Honest gaps', '## History', '## Go deeper'];
var last = -1;
order.forEach(function (h) {
  var i = md.indexOf('\n' + h + '\n');
  assert(i > last, 'v4.3 section out of order or missing: ' + h);
  last = i;
});
['Why v4.', 'retired', 'disagree', 'living cover —', 'claim cover'].forEach(function (s) {
  assert(md.slice(0, md.indexOf('## History')).indexOf(s) === -1, 'v4.3 must not open with meta-talk: ' + s);
});
['**Chat.**', '**Standalone.**', '**Offline.**'].forEach(function (s) { assert(md.indexOf(s) !== -1, 'how to try must cover ' + s); });

/* Claim lock lines. */
assert(md.indexOf('Layer 0 is a shell, not sampler weights.') !== -1, 'stack keeps the Layer 0 lock sentence verbatim');
assert(md.indexOf('AG-SML v1.2') !== -1 && md.indexOf('AG-SML v1.0') === -1, 'licence stays AG-SML v1.2');
assert(md.indexOf('info@Rathor.ai') !== -1, 'contact');
assert(md.indexOf('14.15.6') !== -1, 'workspace identity');
assert(md.indexOf('no** public rathor.ai key proxy') !== -1, 'no key proxy');
assert(md.indexOf('No measured METR numbers') !== -1, 'METR gap stays flagged');
['AGSi demonstrated', 'production-grade', 'superintelligence', 'Superintelligence', 'gap closed'].forEach(function (s) {
  assert(md.indexOf(s) === -1, 'v4.3 must not claim: ' + s);
});

/* History links. */
assert(md.indexOf('/docs/WHITEPAPER_v4.2.html') !== -1, 'history links v4.2');
assert(md.indexOf('archive/root-releases/WHITEPAPER_v4.1.md') !== -1, 'history links v4.1');
assert(md.indexOf('docs/archive') !== -1, 'history links the archive');

/* Page and links. */
assert(html.indexOf('>Read the source on GitHub</a>') !== -1, 'v4.3 page button reads Read the source on GitHub');
assert(html.indexOf('/assets/art/hero-docs-archive-light-1280.webp') !== -1, 'v4.3 keeps the current doc hero art');
assert(fs.existsSync(path.join(root, 'WHITEPAPER_v4.3.md')), 'root pointer exists');
assert(read('index.html').indexOf('data-i18n="paperKicker">Primary technical paper<') !== -1, 'home keeps the Primary technical paper kicker');
assert(read('index.html').indexOf('href="/docs/WHITEPAPER_v4.3.html"') !== -1, 'home links v4.3');
assert(read('sitemap.xml').indexOf('https://rathor.ai/docs/WHITEPAPER_v4.3.html') !== -1, 'sitemap lists v4.3');
assert(read('js/family-nav-2026-08-22.js').indexOf("{ href: '/docs/WHITEPAPER_v4.3.html', label: 'Whitepaper' }") !== -1, 'top nav has Whitepaper');

/* History stays in place. */
['WHITEPAPER_v4.2.md', 'docs/WHITEPAPER_v4.2.md', 'docs/WHITEPAPER_v4.2.html', 'WHITEPAPER_v4.1.md', 'docs/archive/root-releases/WHITEPAPER_v4.1.md'].forEach(function (rel) {
  assert(fs.existsSync(path.join(root, rel)), rel + ' stays as history');
});

console.log('whitepaper-v43-1: ok');
