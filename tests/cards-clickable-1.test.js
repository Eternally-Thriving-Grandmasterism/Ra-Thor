/* CARDS-CLICKABLE-1: owner rule (Sherif, 2026-11-02, permanent) — every link card is
 * fully clickable. The shared .card-link / .stretched-link pattern lives in
 * css/rathor-theme.css, and each card group below uses it with one primary link.
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
assert(/\.card-link\s*\{[^}]*position:\s*relative/.test(css), '.card-link must set position: relative');
assert(/\.card-link\s*\{[^}]*cursor:\s*pointer/.test(css), '.card-link must set cursor: pointer');
assert(/\.stretched-link::after\s*\{[^}]*content:\s*""[^}]*position:\s*absolute[^}]*inset:\s*0/.test(css),
  '.stretched-link::after must cover the card');
assert(/\.card-link-secondary\s*\{[^}]*z-index/.test(css), '.card-link-secondary must lift above the cover');
assert(/\.card-link:has\(\.stretched-link:focus-visible\)/.test(css), 'card must show a focus-visible ring');

// file -> expected number of cards using the pattern
var groups = {
  'index.html': 5,
  'science-watches.html': 2,
  'contact.html': 2,
  'Launch-Ra-Thor.html': 1
};
Object.keys(groups).forEach(function (rel) {
  var html = read(rel);
  var cards = (html.match(/class="[^"]*\bcard-link\b/g) || []).length;
  var links = (html.match(/<a [^>]*class="stretched-link[\s"]/g) || []).length;
  assert(cards === groups[rel], rel + ': expected ' + groups[rel] + ' .card-link cards, found ' + cards);
  assert(links === groups[rel], rel + ': expected one .stretched-link per card, found ' + links);
  assert(/rathor-theme\.css/.test(html), rel + ' must load css/rathor-theme.css');
});

console.log('CARDS-CLICKABLE-1: shared card-link class present and used on ' +
  Object.keys(groups).length + ' pages');
