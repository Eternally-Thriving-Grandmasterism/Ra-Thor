/* PUBLIC-LAUNCH-RETIRE-1: docs/public-launch/index.html is now a minimal English
 * redirect to the Launch map (noindex, canonical, one plain link, no copy). */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
function assert(c, m) { if (!c) throw new Error(m); }
var h = fs.readFileSync(path.join(root, 'docs/public-launch/index.html'), 'utf8');
assert(/^<!DOCTYPE html>/i.test(h), 'has doctype');
assert(h.indexOf('<html lang="en">') !== -1, 'English');
assert(/<title>[^<]+<\/title>/.test(h), 'has title');
assert(h.indexOf('<meta name="robots" content="noindex">') !== -1, 'noindex');
assert(h.indexOf('<link rel="canonical" href="https://rathor.ai/Launch-Ra-Thor.html">') !== -1, 'canonical');
assert(h.indexOf('http-equiv="refresh" content="0; url=/Launch-Ra-Thor.html"') !== -1, 'meta refresh');
assert((h.match(/<a /g) || []).length === 1 && h.indexOf('<a href="/Launch-Ra-Thor.html">') !== -1, 'one plain link');
assert(/^[\x00-\x7F]*$/.test(h), 'ASCII-only, no foreign-language text');
assert(h.length < 600, 'stays minimal (' + h.length + ' bytes)');
assert(fs.existsSync(path.join(root, 'Launch-Ra-Thor.html')), 'target exists');
assert(!/<script|<style|<img/i.test(h), 'no scripts, styles or images');
['sitemap.xml', 'sw.js'].forEach(function (f) {
  assert(fs.readFileSync(path.join(root, f), 'utf8').indexOf('public-launch') === -1, f + ' has no reference');
});
console.log('public-launch-retire-1 ok');
