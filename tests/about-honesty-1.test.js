/* COPY-ABOUT-1: the English-only honesty paragraph keeps the locked Layer 0 line
   and is hidden when a translated pack kicker (which already carries it) is applied. */
var fs = require('fs');
var path = require('path');
var assert = require('assert');
var root = path.join(__dirname, '..');
var html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
var m = html.match(/<p id="rt-honesty"[^>]*>([^<]*)<\/p>/);
assert(m, 'index.html must have p#rt-honesty');
assert(m[1].indexOf('Layer 0 is a shell, not sampler weights.') !== -1, 'locked Layer 0 line must stay verbatim');
assert(m[1].indexOf('METR') !== -1, 'METR caveat must stay');
['js/i18n-chrome.js', 'js/site-lock-2026-08-22.js'].forEach(function (rel) {
  var src = fs.readFileSync(path.join(root, rel), 'utf8');
  assert(/getElementById\('rt-honesty'\)[\s\S]{0,200}\.hidden = /.test(src), rel + ' must hide #rt-honesty for translated kickers');
});
console.log('about-honesty-1.test.js ok');
