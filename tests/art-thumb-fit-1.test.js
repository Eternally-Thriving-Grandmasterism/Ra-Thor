/* ART-THUMB-FIT-1: card thumbnails stay inside the card's content box.
 * Each page that carries thumbnails sizes them from its own <style> block, so a browser that
 * still holds an older cached rathor-theme.css (Pages max-age=600) cannot fall back to the
 * img's 720x315 attributes and run past the card. The theme rule carries the same box. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
function assert(cond, msg) { if (!cond) throw new Error(msg); }
function read(rel) { return fs.readFileSync(path.join(root, rel), 'utf8'); }

var NEED = ['display: block', 'box-sizing: border-box', 'width: 100%', 'max-width: 100%', 'height: auto',
  'aspect-ratio: 16 / 7', 'max-height: 12rem', 'margin: 0 0 1.1rem', 'object-fit: cover', 'border-radius: 1rem',
  'pointer-events: none'];

function ruleBody(css, where) {
  var at = css.indexOf('img.rt-card-thumb {');
  assert(at !== -1, where + ' has an img.rt-card-thumb rule');
  return css.slice(at, css.indexOf('}', at));
}
function checkRule(body, where) {
  NEED.forEach(function (d) { assert(body.indexOf(d) !== -1, where + ' thumb rule needs ' + d); });
  assert(!/margin[^;]*-\d/.test(body), where + ' thumb rule has no negative margin');
  assert(!/(^|[^-])width:\s*\d+px/.test(body), where + ' thumb rule has no fixed pixel width');
}

checkRule(ruleBody(read('css/rathor-theme.css'), 'css/rathor-theme.css'), 'theme');

var PAGES = { 'examples.html': 3, 'Launch-Ra-Thor.html': 5 };
Object.keys(PAGES).forEach(function (page) {
  var html = read(page);
  var head = html.slice(0, html.indexOf('</head>'));
  var styles = (head.match(/<style>[\s\S]*?<\/style>/g) || []).join('\n');
  checkRule(ruleBody(styles, page + ' <style>'), page);
  var tags = html.match(/<img class="rt-card-thumb"[^>]*>/g) || [];
  assert(tags.length === PAGES[page], page + ' keeps ' + PAGES[page] + ' thumbnails');
  tags.forEach(function (t) {
    assert(/\swidth="720"/.test(t) && /\sheight="315"/.test(t), page + ' thumb keeps width/height for layout (16:7)');
    assert(!/\sstyle="/.test(t), page + ' thumb has no inline style attribute');
  });
});

/* Pilot: the content-box wrapper (w-full + px-6) overflowed from 640px to about 830px, which Android
 * Chrome reaches with page zoom or desktop-site mode. HERO-MOBILE-FIT-1 (#54) keeps it border-box up to
 * 831.98px; this card relies on that one rule and adds no second, overlapping wrapper rule. */
var pilot = read('pilot.html');
var pstyle = pilot.slice(pilot.indexOf('<style>'), pilot.indexOf('</style>'));
var wrapperRules = pstyle.match(/body > \.min-h-screen > \.max-w-3xl\s*\{/g) || [];
assert(wrapperRules.length === 1, 'pilot has one wrapper fit rule, got ' + wrapperRules.length);
var fit = pstyle.indexOf('@media (max-width: 831.98px) {');
assert(fit !== -1, 'pilot wrapper fit rule covers widths up to 831.98px');
var fitBody = pstyle.slice(fit, pstyle.indexOf('}', pstyle.indexOf('}', fit) + 1));
assert(fitBody.indexOf('box-sizing: border-box') !== -1 && fitBody.indexOf('max-width: 100%') !== -1, 'pilot wrapper is border-box and capped at 100% up to 831.98px');
assert(pilot.indexOf('<div class="max-w-3xl mx-auto px-4 sm:px-6 py-10 sm:py-14 z-10 w-full">') !== -1, 'pilot wrapper markup unchanged');

console.log('art-thumb-fit-1.test.js ok');
