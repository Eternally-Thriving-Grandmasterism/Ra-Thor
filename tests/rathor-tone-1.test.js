/* RATHOR-TONE-1: one calm body grey, gold titles, underlined prose links.
   Pills and buttons stay shaped and are not underlined. Headings are not underlined.
 */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

var css = read('css/rathor-theme.css');
assert(css.indexOf('--text-calm: #a9a59d;') !== -1, 'dark calm grey is the footer composite');
assert(css.indexOf('--text-calm: #4a3b28;') !== -1, 'light calm grey matches the sand muted brown');
assert(css.indexOf('--rt-muted: var(--text-calm);') !== -1, 'muted aliases the one token');
assert(css.indexOf('--rt-faint: var(--text-calm);') !== -1, 'faint aliases the one token');
assert(css.indexOf('color: var(--text-calm) !important;') !== -1, 'body uses the calm token');
assert(css.indexOf('text-decoration-line: underline !important;') !== -1, 'prose links are underlined');
assert(css.indexOf('text-decoration-thickness: 1px !important;') !== -1, 'resting underline is 1px');
assert(css.indexOf('text-underline-offset: 0.2em !important;') !== -1, 'underline sits below the text');
assert(css.indexOf('text-decoration-thickness: 2px !important;') !== -1, 'hover and focus thicken the underline');
assert(css.indexOf('text-decoration-line: none !important;') !== -1, 'headings and controls can suppress underline');
assert(css.indexOf('#rt-family-nav a,') !== -1, 'family nav is excluded from the underline');
assert(css.indexOf('.rt-follow-list a,') !== -1, 'follow row is excluded so its border is not doubled');
assert(css.indexOf('a.rt-btn,') !== -1, 'button links are excluded from the underline');
assert(css.indexOf('a[class*="rounded-full"]') !== -1, 'pill links are excluded from the underline');
assert(css.indexOf('text-transform: uppercase !important;') !== -1, 'section headings take the footer label case');
assert(css.indexOf('letter-spacing: 0.04em !important;') !== -1, 'titles use the footer tracking');
assert(css.indexOf('outline: 2px solid var(--rt-gold-hot);') !== -1, 'keyboard focus keeps a visible ring');
assert(css.indexOf('.message.user a') !== -1, 'user-bubble links stay readable on gold');

var pilot = read('pilot.html');
assert(pilot.indexOf('color: var(--text-calm);') !== -1, 'pilot body points at the token');
assert(pilot.indexOf('#f6f1e4') === -1, 'pilot does not keep a cream hex fallback');
assert(pilot.indexOf('rgba(246, 241, 228, 0.68)') === -1, 'pilot does not keep a muted rgba fallback');

var offline = read('offline.html');
assert(offline.indexOf('color: var(--text-calm);') !== -1, 'offline body points at the token');
assert(offline.indexOf('rgba(246,241,228,0.46)') === -1, 'offline note no longer uses the faint grey');

assert(read('sw.js').indexOf('--text-calm') === -1, 'service worker is untouched by the tone token');
assert(read('js/family-nav-2026-08-22.js').indexOf('--text-calm') === -1, 'family nav script is untouched');

console.log('rathor-tone-1.test.js ok');
