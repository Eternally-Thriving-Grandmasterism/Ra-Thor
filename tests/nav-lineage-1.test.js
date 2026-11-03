// NAV-LINEAGE-1: the shared top nav (family-nav) lists Lineage and Examples,
// every public page loads family-nav, and en has a navLineage key.
'use strict';
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const root = path.join(__dirname, '..');
const nav = fs.readFileSync(path.join(root, 'js/family-nav-2026-08-22.js'), 'utf8');
const links = nav.slice(nav.indexOf('var LINKS = ['), nav.indexOf('];', nav.indexOf('var LINKS = [')));
assert.ok(/href: '\/lineage\.html'/.test(links), 'top nav LINKS must include /lineage.html');
assert.ok(/href: '\/examples\.html'/.test(links), 'top nav LINKS must include /examples.html');
for (const gone of ['Launch-Ra-Thor', 'sovereign-shard', 'web-forge']) {
  assert.ok(links.indexOf('/' + gone + '.html') === -1, gone + ' lives under Examples, not the top bar');
  assert.ok(nav.indexOf('<a href="/' + gone + '.html"') !== -1, gone + ' stays in the fallback footer');
  assert.ok(fs.readFileSync(path.join(root, 'examples.html'), 'utf8').indexOf('href="/' + gone + '.html"') !== -1, 'examples.html links ' + gone);
}
assert.ok(nav.indexOf('<a href="/lineage.html">Lineage</a>') !== -1, 'fallback footer links Lineage');
assert.ok(nav.indexOf('<a href="/examples.html">Examples</a>') !== -1, 'fallback footer links Examples');
const PUBLIC = ['index.html', 'chat.html', 'employ.html', 'pilot.html', 'Launch-Ra-Thor.html',
  'micro-moment.html', 'sovereign-shard.html', 'web-forge.html', 'contact.html', 'privacy.html',
  'licence.html', 'lineage.html', 'examples.html', 'science-watches.html', 'constellation-week.html', 'briefing.html', 'go-x.html'];
for (const f of PUBLIC) {
  const html = fs.readFileSync(path.join(root, f), 'utf8');
  assert.ok(/\/js\/family-nav-2026-08-22\.js/.test(html), f + ' must load family-nav (top nav with Lineage)');
}
const en = fs.readFileSync(path.join(root, 'i18n/en.js'), 'utf8');
assert.ok(/"navLineage": "Lineage"/.test(en), 'en navLineage key');
assert.ok(/"navExamples": "Examples"/.test(en), 'en navExamples key');
console.log('nav-lineage-1: ok');
