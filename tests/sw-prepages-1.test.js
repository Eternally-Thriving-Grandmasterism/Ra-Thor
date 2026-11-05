/* SW-PREPAGES-1: precache the public nav pages added after NAV-LINEAGE-1,
 * and keep the worker cache LOCK in step with that list.
 * Icon entries stay on the real files already in icons/.
 * The worker is evaluated in a stub scope so install and navigation
 * are checked as behavior, not only as source text. */
var fs = require('fs');
var path = require('path');
var vm = require('vm');

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

var sw = read('sw.js');

assert(sw.indexOf('LOCK 20261105a') !== -1, 'header LOCK must be 20261105a');
assert(sw.indexOf("var LOCK = '20261105a';") !== -1, 'worker cache LOCK must be 20261105a');
assert(sw.indexOf("var LOCK = '20261022f';") === -1, 'previous worker cache LOCK must not stay assigned');
assert(sw.indexOf("var CACHE = 'rathor-core-' + LOCK;") !== -1, 'CACHE name must stay rathor-core- plus LOCK');
assert(sw.indexOf('i18n pack token 20260924a') !== -1, 'i18n pack token stays 20260924a');
assert(sw.indexOf('ignoreSearch: true') !== -1, 'offline i18n loads still ignore the cache query');
assert(sw.indexOf("pathname.indexOf('/i18n/')") !== -1, 'offline i18n fallback still covers /i18n/');

var navAt = sw.indexOf("if (req.mode === 'navigate' || req.destination === 'document')");
var scriptAt = sw.indexOf("if (req.destination === 'script' || req.destination === 'style')");
assert(navAt !== -1 && scriptAt > navAt, 'document handler stays ahead of the script handler');
var navBlock = sw.slice(navAt, scriptAt);
var fetchAt = navBlock.indexOf('fetch(req)');
var cacheAt = navBlock.indexOf('caches.match(req)');
assert(fetchAt !== -1 && cacheAt > fetchAt, 'documents stay network-first; cache is only the offline fallback');

var block = sw.match(/var\s+PRECACHE\s*=\s*\[([\s\S]*?)\];/);
assert(block, 'could not find PRECACHE list in sw.js');
var urls = [];
var sre = /'([^']+)'|"([^"]+)"/g;
var s;
while ((s = sre.exec(block[1]))) urls.push(s[1] || s[2]);

['/examples.html', '/lineage.html'].forEach(function (page) {
  assert(urls.indexOf(page) !== -1, 'PRECACHE must list ' + page);
  assert(isFile(page.slice(1)), page + ' must exist in the repo');
});

var icons = urls.filter(function (u) { return u.indexOf('/icons/') === 0; });
assert(icons.indexOf('/icons/ra-thor-icon-192.png') !== -1, 'PRECACHE must list the 192 icon');
assert(icons.indexOf('/icons/ra-thor-icon-512.png') !== -1, 'PRECACHE must list the 512 icon');
icons.forEach(function (icon) {
  assert(isFile(icon.slice(1)), 'precached icon missing on disk: ' + icon);
});

function linkedIcons(html) {
  var out = [];
  var re = /<link\b[^>]*>/gi;
  var m;
  while ((m = re.exec(html))) {
    var tag = m[0];
    if (!/\brel=["'](?:shortcut icon|icon|apple-touch-icon)["']/i.test(tag)) continue;
    var href = tag.match(/\bhref=["']([^"']+)["']/i);
    if (!href) continue;
    var url = href[1].split('?')[0].split('#')[0];
    if (url.charAt(0) !== '/') continue;
    out.push(url);
  }
  return out;
}

urls.forEach(function (u) {
  if (u !== '/' && !/\.html$/.test(u)) return;
  var rel = u === '/' ? 'index.html' : u.slice(1);
  linkedIcons(read(rel)).forEach(function (icon) {
    assert(isFile(icon.slice(1)), rel + ' links icon missing on disk: ' + icon);
    assert(urls.indexOf(icon) !== -1, rel + ' icon must stay in PRECACHE: ' + icon);
  });
});

var missing = urls.filter(function (u) {
  var rel = u === '/' ? 'index.html' : u.replace(/^\//, '');
  return !isFile(rel);
});
assert(missing.length === 0, 'PRECACHE paths missing in repo: ' + missing.join(', '));

var added = [];
var matchCalls = [];
var handlers = {};
var offline = false;
var sandbox = {
  console: console,
  URL: URL,
  self: {
    location: { origin: 'https://rathor.ai' },
    skipWaiting: function () {},
    clients: { claim: function () { return Promise.resolve(); } },
    addEventListener: function (type, fn) { handlers[type] = fn; }
  },
  caches: {
    open: function (name) {
      return Promise.resolve({
        add: function (url) {
          added.push(name + ' ' + url);
          return Promise.resolve();
        },
        put: function () { return Promise.resolve(); }
      });
    },
    match: function (req) {
      var key = typeof req === 'string' ? req : (req && req.url) || '';
      matchCalls.push(key);
      return Promise.resolve({ url: key, from: 'cache' });
    },
    keys: function () { return Promise.resolve([]); }
  },
  fetch: function (req) {
    if (offline) return Promise.reject(new Error('offline'));
    return Promise.resolve({ url: req.url, from: 'network', ok: true });
  }
};
vm.createContext(sandbox);
vm.runInContext(sw, sandbox, { filename: 'sw.js' });

assert(sandbox.LOCK === '20261105a', 'evaluated LOCK must be 20261105a');
assert(sandbox.CACHE === 'rathor-core-20261105a', 'evaluated cache name must include the new LOCK');
['/examples.html', '/lineage.html'].forEach(function (page) {
  assert(sandbox.PRECACHE.indexOf(page) !== -1, 'evaluated PRECACHE must list ' + page);
});

var installWait;
handlers.install({ waitUntil: function (p) { installWait = p; } });
installWait.then(function () {
  assert(added.indexOf('rathor-core-20261105a /examples.html') !== -1, 'install must precache /examples.html');
  assert(added.indexOf('rathor-core-20261105a /lineage.html') !== -1, 'install must precache /lineage.html');
  assert(added.indexOf('rathor-core-20261105a /icons/ra-thor-icon-192.png') !== -1, 'install must precache the 192 icon');
  assert(added.indexOf('rathor-core-20261105a /icons/ra-thor-icon-512.png') !== -1, 'install must precache the 512 icon');

  function navigate(url) {
    var responded;
    handlers.fetch({
      request: { method: 'GET', mode: 'navigate', destination: 'document', url: url },
      respondWith: function (p) { responded = p; }
    });
    return responded;
  }

  matchCalls.length = 0;
  return navigate('https://rathor.ai/examples.html').then(function (res) {
    assert(res.from === 'network', 'online navigation must use the network response');
    assert(matchCalls.length === 0, 'online navigation must not read the cache first');
    offline = true;
    matchCalls.length = 0;
    return navigate('https://rathor.ai/lineage.html');
  }).then(function (res) {
    assert(res.from === 'cache', 'offline navigation must fall back to the cache');
    assert(matchCalls[0] === 'https://rathor.ai/lineage.html', 'offline fallback must match the requested document first');
    console.log('sw-prepages-1: LOCK 20261105a; PRECACHE lists /examples.html and /lineage.html; ' + icons.length + ' icon paths exist; install adds ' + added.length + ' urls');
  });
}).catch(function (err) {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
