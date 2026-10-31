/* INSTALL-HINTS-2: per-browser install steps in js/pwa-install.js.
 * Runs the script in a vm with a tiny fake DOM (no jsdom in this repo). */
var fs = require('fs');
var path = require('path');
var vm = require('vm');
var SRC = fs.readFileSync(path.join(__dirname, '..', 'js', 'pwa-install.js'), 'utf8');

function assert(cond, msg) { if (!cond) throw new Error(msg); }

function el(tag) {
  var e = { tagName: tag, id: '', attrs: {}, style: {}, children: [], parentNode: null, listeners: {}, innerHTML: '', textContent: '',
    setAttribute: function (k, v) { this.attrs[k] = String(v); if (k === 'id') this.id = v; },
    getAttribute: function (k) { return k in this.attrs ? this.attrs[k] : null; },
    addEventListener: function (t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    appendChild: function (c) { c.parentNode = this; this.children.push(c); return c; },
    removeChild: function (c) { this.children.splice(this.children.indexOf(c), 1); c.parentNode = null; },
    querySelector: function () { return null; }, querySelectorAll: function () { return []; } };
  return e;
}

function load(opts) {
  var timers = [], winL = {}, local = {}, session = {};
  var body = el('body');
  function store(m) { return { getItem: function (k) { return k in m ? m[k] : null; }, setItem: function (k, v) { m[k] = String(v); } }; }
  var document = { body: body, readyState: 'complete', referrer: '',
    createElement: el, addEventListener: function () {},
    getElementById: function (id) { return body.children.filter(function (c) { return c.id === id; })[0] || null; },
    querySelectorAll: function () { return []; } };
  var nav = { userAgent: opts.ua, platform: opts.platform || '', maxTouchPoints: opts.touch || 0 };
  var win = { addEventListener: function (t, f) { (winL[t] = winL[t] || []).push(f); },
    matchMedia: function () { return { matches: false }; }, navigator: nav };
  win.self = win; win.top = win;
  var ctx = { window: win, document: document, navigator: nav, localStorage: store(local), sessionStorage: store(session),
    setTimeout: function (f, ms) { timers.push({ f: f, ms: ms }); }, Date: Date, parseInt: parseInt, String: String, isNaN: isNaN };
  vm.runInNewContext(SRC, ctx);
  return { win: win, body: body, local: local, session: session,
    runTimers: function (ms) { var due = timers.filter(function (t) { return t.ms <= ms; }); timers = timers.filter(function (t) { return t.ms > ms; }); due.forEach(function (t) { t.f(); }); },
    fire: function (t, ev) { (winL[t] || []).forEach(function (f) { f(ev); }); },
    banner: function () { return body.children.filter(function (c) { return c.id === 'rathor-pwa-banner'; })[0] || null; } };
}

var UA = {
  chrome: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
  ffwin: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0',
  ffmac: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:150.0) Gecko/20100101 Firefox/150.0',
  fflinux: 'Mozilla/5.0 (X11; Linux x86_64; rv:150.0) Gecko/20100101 Firefox/150.0',
  ffandroid: 'Mozilla/5.0 (Android 14; Mobile; rv:150.0) Gecko/150.0 Firefox/150.0',
  safarimac: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15',
  iphone: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1',
  crios: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/141.0 Mobile/15E148 Safari/604.1',
  samsung: 'Mozilla/5.0 (Linux; Android 14; SM-S921B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/28.0 Chrome/130.0.0.0 Mobile Safari/537.36',
  edgea: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36 EdgA/141.0',
  opera: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36 OPR/90.0'
};
var n = 0;
function kindOf(ua, extra) { return load(Object.assign({ ua: ua }, extra || {})).win.__rtPwa.kind(); }

// 1. UA detection.
[['chrome', 'chromium'], ['ffwin', 'firefox-win'], ['ffmac', 'firefox-nowin'], ['fflinux', 'firefox-nowin'],
 ['ffandroid', 'firefox-android'], ['safarimac', 'safari-mac'], ['iphone', 'ios'], ['crios', 'ios'],
 ['samsung', 'samsung'], ['edgea', 'edge-android'], ['opera', 'opera-android']].forEach(function (c) {
  var k = kindOf(UA[c[0]]); assert(k === c[1], c[0] + ': expected ' + c[1] + ', got ' + k); n++;
});
assert(kindOf(UA.safarimac, { platform: 'MacIntel', touch: 5 }) === 'ios', 'iPadOS desktop UA must be ios'); n++;
assert(kindOf(UA.safarimac, { platform: 'MacIntel', touch: 0 }) === 'safari-mac', 'Mac without touch stays safari-mac'); n++;

// 2. Copy chosen on tap.
function tapCopy(ua, extra) { var w = load(Object.assign({ ua: ua }, extra || {})); w.win.rathorTriggerPWAInstall(); var b = w.banner(); return b ? b.innerHTML : ''; }
[['ffwin', 'Add tab to taskbar'], ['safarimac', 'File → Add to Dock'], ['iphone', 'Add to Home Screen'],
 ['samsung', 'Add page to → Home screen'], ['edgea', 'Add to phone'], ['opera', 'Add to → Home screen'],
 ['ffandroid', 'Add app to Home screen'], ['chrome', 'Chrome install']].forEach(function (c) {
  var h = tapCopy(UA[c[0]]); assert(h.indexOf(c[1]) !== -1, c[0] + ': copy missing "' + c[1] + '"'); n++;
});
assert(tapCopy(UA.iphone).indexOf('Open as Web App') !== -1, 'iOS copy must mention Open as Web App'); n++;
var nowin = tapCopy(UA.ffmac);
assert(nowin.indexOf('can’t install web apps yet') !== -1 && nowin.indexOf('Bookmark') !== -1, 'Firefox macOS: honest fallback'); n++;
assert(nowin.indexOf('rathor-pwa-install-btn') === -1, 'Firefox macOS: no fake install button'); n++;
assert(tapCopy(UA.ffwin).indexOf('Firefox on desktop can’t') === -1, 'Firefox Windows: no false can’t-install copy'); n++;

// 3. Auto hint does not write the 14-day dismiss key; late BIP replaces it.
var w = load({ ua: UA.samsung });
w.runTimers(5000);
assert(w.banner() && w.banner().getAttribute('data-kind') === 'hint', 'auto hint shown for Samsung'); n++;
assert(!('rathor-pwa-install-dismissed' in w.local), 'auto hint must not set dismiss flag'); n++;
w.fire('beforeinstallprompt', { preventDefault: function () {}, prompt: function () {} });
var b = w.banner();
assert(b && b.getAttribute('data-kind') !== 'hint', 'late BIP must replace the hint with the real banner'); n++;
assert(b.innerHTML.indexOf('rathor-pwa-install-btn') !== -1, 'real banner has the install button'); n++;
assert(w.body.children.filter(function (c) { return c.id === 'rathor-pwa-banner'; }).length === 1, 'only one banner'); n++;
w.runTimers(20000);
assert(w.banner() === b, 'hint timeout must not hide the real banner'); n++;

// 4. Chromium default: no auto hint.
var c = load({ ua: UA.chrome }); c.runTimers(5000);
assert(!c.banner(), 'Chrome waits for BIP, no auto hint'); n++;

console.log('pwa-install-hints: ' + n + ' checks passed');
