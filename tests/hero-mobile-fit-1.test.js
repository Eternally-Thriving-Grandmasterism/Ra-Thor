/* HERO-MOBILE-FIT-1: pilot.html keeps its padded wrapper inside the viewport at every
 * width where width 100% plus side padding would overflow (up to 48rem + 48px + 16px).
 * Before this card the rule stopped at 639px, so a 640-831px layout (Chrome Android
 * with page zoom below 100%) clipped the hero and cards on the right. Static check. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
function assert(cond, msg) { if (!cond) throw new Error(msg); }
var html = fs.readFileSync(path.join(root, 'pilot.html'), 'utf8');
var css = (html.match(/<style[^>]*>([\s\S]*?)<\/style>/i) || [])[1] || '';
var at = css.indexOf('@media (max-width: 831.98px) {');
assert(at !== -1, 'pilot.html: wrapper fit rule must cover widths up to 831.98px');
assert(css.indexOf('@media (max-width: 639px) {') === -1, 'pilot.html: the old 639px-only wrapper rule must not come back');
var block = css.slice(at, css.indexOf('\n    }', at));
assert(/body > \.min-h-screen > \.max-w-3xl \{\s*box-sizing: border-box;\s*max-width: 100%;\s*\}/.test(block),
  'pilot.html: wrapper must be border-box with max-width 100% inside the fit rule');
assert(/<body[^>]*>\s*<div class="min-h-screen[^"]*">\s*<div class="max-w-3xl mx-auto px-4 sm:px-6[^"]*\bw-full\b/.test(html),
  'pilot.html: wrapper markup no longer matches the fit rule selector');
assert(/figure\.rt-art-hero img \{[\s\S]*?width: 100%;[\s\S]*?max-width: 100%;[\s\S]*?height: auto;/.test(
  fs.readFileSync(path.join(root, 'css/rathor-theme.css'), 'utf8')), 'shared hero img sizing must stay width/max-width 100%, height auto');
console.log('HERO-MOBILE-FIT-1: pilot wrapper stays inside the frame up to 831.98px');
