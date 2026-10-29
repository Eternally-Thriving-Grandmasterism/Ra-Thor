/* PHONE-CLIP-1: stop right-edge clipping on phones for /employ.html, /briefing.html
 * and /web-forge.html, without reaching /pilot.html (the Steward: "Do not touch /pilot.html").
 * Static checks only: the scoped rules exist, and none of them can match pilot.html. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}
function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}
function inlineStyles(html) {
  var out = [];
  var re = /<style[^>]*>([\s\S]*?)<\/style>/gi;
  var m;
  while ((m = re.exec(html))) out.push(m[1]);
  return out.join('\n');
}

var WRAP = 'body > .min-h-screen > .max-w-3xl.w-full';
var MEDIA = '@media (max-width: 51.99rem)';

// 1. employ + briefing: page-scoped rule in the page's own <style> block.
['employ.html', 'briefing.html'].forEach(function (f) {
  var html = read(f);
  var css = inlineStyles(html);
  var at = css.indexOf(MEDIA);
  assert(at !== -1, f + ': phone media rule missing');
  var block = css.slice(at, css.indexOf('\n    }', at));
  assert(block.indexOf(WRAP + ' { box-sizing: border-box; }') !== -1, f + ': wrapper box-sizing rule missing');
  assert(block.indexOf(WRAP + ' :is(a, code) { overflow-wrap: anywhere; }') !== -1, f + ': long-token wrap rule missing');
  // The selector must actually match this page's wrapper.
  assert(/<body[^>]*>\s*<div class="min-h-screen[^"]*">\s*<div class="max-w-3xl[^"]*\bw-full\b/.test(html), f + ': wrapper markup no longer matches ' + WRAP);
});

// 2. pilot.html: nothing from this card, and no .w-full / box-sizing override of its own.
var pilot = read('pilot.html');
var pilotCss = inlineStyles(pilot);
assert(pilot.indexOf('PHONE-CLIP-1') === -1, 'pilot.html must not carry PHONE-CLIP-1 rules');
assert(pilotCss.indexOf(MEDIA) === -1, 'pilot.html must not get the phone media rule');
assert(!/\.w-full[^{]*\{[^}]*box-sizing/.test(pilotCss), 'pilot.html must not set box-sizing on .w-full');
assert(!/overflow-wrap:\s*anywhere/.test(pilotCss), 'pilot.html must not get the long-token wrap rule');
assert(!/\brt-forge-grid\b/.test(pilot) && !/\brt-gate-row\b/.test(pilot), 'forge selectors must not match pilot.html');

// 3. Shared CSS stays untouched by this card: no global .w-full box-sizing rule that would reach pilot.
var cssDir = path.join(root, 'css');
fs.readdirSync(cssDir).filter(function (n) { return /\.css$/.test(n); }).forEach(function (n) {
  var s = fs.readFileSync(path.join(cssDir, n), 'utf8');
  assert(!/\.w-full[^{]*\{[^}]*box-sizing/.test(s), 'css/' + n + ' must not add a global .w-full box-sizing rule');
  assert(s.indexOf('PHONE-CLIP-1') === -1, 'css/' + n + ': PHONE-CLIP-1 rules belong in the page files, not shared CSS');
});
var restB = read('css/rathor-theme-rest-b.css');
assert(/^\.w-full \{ width: 100%; \}$/m.test(restB), '.w-full utility must stay unchanged');
assert(restB.indexOf('@media (min-width: 900px) {\n  .rt-forge-grid { grid-template-columns: 1fr 1fr; }') !== -1,
  'shared 900px two-column forge rule must stay');
assert(restB.indexOf('.rt-preset-row {\n  grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));') !== -1,
  'shared .rt-preset-row (Shard) must stay unchanged');

// 4. web-forge: rules live in web-forge.html's own <style>, scoped under .rt-forge-grid.
var forge = read('web-forge.html');
var head = forge.slice(0, forge.indexOf('</head>'));
var forgeCss = inlineStyles(head);
assert(head.indexOf('/css/rathor-theme.css') < head.indexOf('PHONE-CLIP-1'), 'forge <style> must follow the shared stylesheet');
assert(forgeCss.indexOf('@media (max-width: 899.98px) { .rt-forge-grid { grid-template-columns: minmax(0, 1fr); } }') !== -1,
  'forge single-column minmax(0,1fr) rule missing (must stay below the shared 900px rule)');
assert(forgeCss.indexOf('.rt-forge-grid .rt-preset-row { grid-template-columns: repeat(auto-fit, minmax(min(9.5rem, 100%), 1fr)); }') !== -1,
  'forge preset row min() rule missing');
assert(forgeCss.indexOf('.rt-forge-grid .rt-gate-row input[type="range"] { min-width: 0; }') !== -1, 'forge slider min-width rule missing');
assert(/class="rt-forge-grid"[\s\S]*class="rt-preset-row"/.test(forge), 'web-forge.html must keep .rt-preset-row inside .rt-forge-grid');

// 5. Only web-forge.html uses .rt-forge-grid among top-level pages.
fs.readdirSync(root).filter(function (n) { return /\.html$/.test(n); }).forEach(function (n) {
  if (n === 'web-forge.html') return;
  assert(!/\brt-forge-grid\b/.test(read(n)), n + ' unexpectedly uses .rt-forge-grid');
});

console.log('PHONE-CLIP-1 scoped rules present; none match pilot.html');
