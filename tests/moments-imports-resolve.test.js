/* MOMENTS-FIX-1: Moments (/micro-moment.html) and the GPU demo import
 * mercy-motion-vision-engine.js, which imports ./fuzzy-mercy-logic.js from the
 * repo root. That file was archived to js/archive/root-engines/ and the pages
 * died on a 404. This test keeps every relative import in the Moments chain
 * resolvable, and every path in sw.js's PRECACHE list present in the repo.
 * sw.js is only read, never changed. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}
function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}
function isFile(rel) {
  try { return fs.statSync(path.join(root, rel)).isFile(); } catch (e) { return false; }
}

// Static `import ... from '...'`, bare `import '...'`, `export ... from '...'`, dynamic import('...').
function specifiers(src) {
  var out = [];
  var re = /(?:^|[;\s])(?:import|export)\s+(?:[^'"`;]*?\s+from\s+)?['"]([^'"]+)['"]|\bimport\(\s*['"]([^'"]+)['"]\s*\)/gm;
  var m;
  while ((m = re.exec(src))) out.push(m[1] || m[2]);
  return out;
}
function moduleScripts(html) {
  var out = [];
  var re = /<script[^>]*type=["']module["'][^>]*>([\s\S]*?)<\/script>/gi;
  var m;
  while ((m = re.exec(html))) out.push(m[1]);
  return out.join('\n');
}

// Walk the import graph from each entry; every relative/absolute specifier must be a repo file.
var seen = {};
function walk(rel, src, from) {
  specifiers(src).forEach(function (spec) {
    if (!/^(\.\.?\/|\/)/.test(spec)) return; // skip bare/URL specifiers
    var target = spec.charAt(0) === '/'
      ? spec.slice(1)
      : path.posix.normalize(path.posix.join(path.posix.dirname(rel), spec));
    assert(isFile(target), from + ' imports ' + spec + ' -> ' + target + ' (missing in repo)');
    if (seen[target]) return;
    seen[target] = true;
    if (/\.m?js$/.test(target)) walk(target, read(target), target);
  });
}

var entries = [
  ['mercy-motion-vision-engine.js', read('mercy-motion-vision-engine.js')],
  ['one-organism-launch.js', read('one-organism-launch.js')],
  ['micro-moment.html', moduleScripts(read('micro-moment.html'))],
  ['demos/gpu-micro-moment-demo.html', moduleScripts(read('demos/gpu-micro-moment-demo.html'))]
];
entries.forEach(function (e) { walk(e[0], e[1], e[0]); });

// The known chain must actually be covered.
assert(specifiers(read('mercy-motion-vision-engine.js')).indexOf('./fuzzy-mercy-logic.js') !== -1,
  'engine still expected to import ./fuzzy-mercy-logic.js');
['mercy-motion-vision-engine.js', 'fuzzy-mercy-logic.js', 'one-organism-launch.js'].forEach(function (f) {
  assert(seen[f], 'import walk did not reach ' + f);
});

// Root fuzzy-mercy-logic.js stays a leaf (no imports) and exports fuzzyMercy.
var fuzzy = read('fuzzy-mercy-logic.js');
assert(specifiers(fuzzy).length === 0, 'fuzzy-mercy-logic.js must not import anything');
assert(/export\s*\{\s*fuzzyMercy\s*\}/.test(fuzzy), 'fuzzy-mercy-logic.js must export fuzzyMercy');
// If the archive copy is still kept, the root file must match it byte for byte.
if (isFile('js/archive/root-engines/fuzzy-mercy-logic.js')) {
  assert(fuzzy === read('js/archive/root-engines/fuzzy-mercy-logic.js'),
    'root fuzzy-mercy-logic.js must stay byte-identical to js/archive/root-engines/fuzzy-mercy-logic.js');
}

// Every path in sw.js PRECACHE exists (read-only parse).
var sw = read('sw.js');
var block = sw.match(/var\s+PRECACHE\s*=\s*\[([\s\S]*?)\];/);
assert(block, 'could not find PRECACHE list in sw.js');
var urls = [];
var sre = /'([^']+)'|"([^"]+)"/g;
var s;
while ((s = sre.exec(block[1]))) urls.push(s[1] || s[2]);
assert(urls.length > 20, 'PRECACHE parse looks wrong: ' + urls.length + ' entries');
assert(urls.indexOf('/fuzzy-mercy-logic.js') !== -1, 'PRECACHE expected to list /fuzzy-mercy-logic.js');
var missing = urls.filter(function (u) {
  var rel = u === '/' ? 'index.html' : u.replace(/^\//, '');
  return !isFile(rel);
});
assert(missing.length === 0, 'sw.js PRECACHE paths missing in repo: ' + missing.join(', '));

console.log('Moments imports resolve: ' + Object.keys(seen).length + ' modules; sw.js PRECACHE ' + urls.length + ' paths all present');
