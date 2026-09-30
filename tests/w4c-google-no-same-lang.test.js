/* W4c (TRANSLATE-FIX-1): never build an English-to-English Google Translate link.
 * translate.google.com/website?sl=en&u=... takes the target from the UI language,
 * so an English phone lands on rathor-ai.translate.goog?_x_tr_sl=en&_x_tr_tl=en,
 * which Google answers with HTTP 400 "Can't translate this page".
 * With no non-English target the strip stays (same height) and shows a hint
 * pointing at the site's own language buttons. */
var fs = require('fs');
var path = require('path');
var vm = require('vm');
var root = path.join(__dirname, '..');
var src = fs.readFileSync(path.join(root, 'js/google-translate-optin.js'), 'utf8');
var enSrc = fs.readFileSync(path.join(root, 'i18n/en.js'), 'utf8');
var nav = fs.readFileSync(path.join(root, 'js/family-nav-2026-08-22.js'), 'utf8');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

var PACKS = ['ar', 'es', 'fr', 'nl', 'de', 'zh', 'ja', 'pt', 'ru', 'hi', 'it', 'ko',
  'uk', 'pl', 'tr', 'vi', 'id', 'sv', 'th', 'el', 'fa', 'he'];

function loadEn() {
  var sb = { window: { translations: {} } };
  vm.createContext(sb);
  vm.runInContext(enSrc, sb);
  return sb.window.translations.en;
}
var EN = loadEn();

function run(opts) {
  var store = Object.assign({}, opts.store || {});
  var els = {};
  function el(tag) {
    return {
      tagName: tag, attrs: {}, style: {}, hidden: false, textContent: '', className: '',
      setAttribute: function (k, v) { this.attrs[k] = String(v); },
      getAttribute: function (k) { return k in this.attrs ? this.attrs[k] : null; },
      removeAttribute: function (k) { delete this.attrs[k]; },
      set innerHTML(html) {
        var a = el('a');
        a.attrs.target = '_blank';
        a.attrs.rel = 'noopener';
        els['rt-gtranslate-open'] = a;
        els['rt-gtranslate-note'] = el('p');
      }
    };
  }
  if (opts.langSelector !== false) els['lang-selector'] = el('div');
  var body = { firstChild: null, insertBefore: function (n) { if (n.id) els[n.id] = n; } };
  var doc = {
    body: body,
    getElementById: function (id) { return els[id] || null; },
    createElement: function (t) { return el(t); },
    addEventListener: function () {}
  };
  var langs = opts.languages || ['en-US'];
  var win = {
    document: doc,
    localStorage: {
      getItem: function (k) { return k in store ? store[k] : null; },
      setItem: function (k, v) { store[k] = String(v); }
    },
    navigator: { languages: langs, language: langs[0] },
    location: { pathname: opts.path || '/', hostname: opts.host || 'rathor.ai' },
    translations: { en: EN }
  };
  win.window = win;
  vm.runInNewContext(src, win);
  return { win: win, els: els, store: store };
}

function tlOf(href) {
  var m = /[?&]tl=([^&]+)/.exec(href || '');
  return m ? decodeURIComponent(m[1]) : null;
}

// 0. Static guards.
assert(src.indexOf('translate.google.com/website') === -1, 'must not use the /website picker (resolves to tl=en on English phones)');
assert(src.indexOf('tl=en') === -1, 'source must never build tl=en');
assert(src.indexOf('window.rtGTranslateHref') !== -1, 'must expose window.rtGTranslateHref for tests');
assert(typeof EN.gTranslateHint === 'string' && EN.gTranslateHint.trim() !== '', 'en.js must carry gTranslateHint');
assert(src.indexOf("pack('gTranslateHint'") !== -1, 'hint must come from the pack via pack()');
assert(nav.indexOf('google-translate-optin.js?v=20260915c') === -1, 'optin cache tag must be bumped');

// 1. English page, English-only device, nothing stored: no Google href, hint shown, strip kept.
var r = run({ store: { 'rathor-lang': 'en' }, languages: ['en-US', 'en'] });
assert(r.win.rtGTranslateHref('en') === '', 'en + English-only device must not build a Google link');
var wrap = r.els['rt-gtranslate'];
assert(wrap, 'strip must still mount for an English-only device');
assert(wrap.hidden === false && wrap.style.display !== 'none', 'strip must not be hidden (landing layout unchanged)');
var a = r.els['rt-gtranslate-open'];
assert(a.getAttribute('href') === '#lang-selector', 'hint must anchor to the site language buttons, got ' + a.getAttribute('href'));
assert(a.textContent === EN.gTranslateHint, 'hint text must come from en.gTranslateHint');
assert(a.getAttribute('target') === null, 'in-page hint must not open a new tab');
assert(a.getAttribute('data-rt-gtranslate-hint') === '1', 'hint state must be marked');
assert(!/google/i.test(a.getAttribute('href') || ''), 'no Google href in hint state');

// 1b. No language buttons on the page: hint stays, no href at all.
r = run({ store: { 'rathor-lang': 'en' }, languages: ['en-GB'], langSelector: false });
assert(r.els['rt-gtranslate-open'].getAttribute('href') === null, 'hint without a language bar must not carry an href');
assert(r.els['rt-gtranslate-open'].textContent === EN.gTranslateHint, 'hint text still shown without a language bar');

// 2. English page, device prefers Spanish second: tl=es and the button is a Google new tab.
r = run({ store: { 'rathor-lang': 'en' }, languages: ['en-US', 'es-ES'] });
assert(tlOf(r.win.rtGTranslateHref('en')) === 'es', 'en + es device must target es');
a = r.els['rt-gtranslate-open'];
assert(tlOf(a.getAttribute('href')) === 'es' && a.getAttribute('target') === '_blank' && a.getAttribute('rel') === 'noopener', 'es button must be a Google new tab');
assert(a.textContent === EN.gTranslateBtn, 'button text must be gTranslateBtn');

// 2b. English page, remembered non-English language wins over the device list.
r = run({ store: { 'rathor-lang': 'en', 'rathor-gtranslate-tl': 'ja' }, languages: ['en-US', 'es'] });
assert(tlOf(r.win.rtGTranslateHref('en')) === 'ja', 'remembered ja must win');

// 3. Every non-English pack targets itself, source en, u=https://rathor.ai/...
PACKS.forEach(function (l) {
  var x = run({ store: { 'rathor-lang': l }, path: '/employ.html' }).win.rtGTranslateHref(l);
  assert(x === 'https://translate.google.com/translate?sl=en&tl=' + encodeURIComponent(l) +
    '&u=' + encodeURIComponent('https://rathor.ai/employ.html'), l + ' link shape: ' + x);
});

// 4. Never an English target, whatever the device says.
[['en-US'], ['en-GB', 'en'], ['en'], ['en-AU', 'en-CA'], []].forEach(function (langs) {
  var x = run({ store: { 'rathor-lang': 'en', 'rathor-gtranslate-tl': 'en' }, languages: langs }).win.rtGTranslateHref('en');
  assert(x === '', 'no link for English-only device ' + JSON.stringify(langs) + ', got ' + x);
});

// 5. Under Google's proxy: do not mount at all.
['rathor-ai.translate.goog', 'RATHOR-AI.TRANSLATE.GOOG'].forEach(function (host) {
  var p = run({ store: { 'rathor-lang': 'es' }, host: host });
  assert(!p.els['rt-gtranslate'] && !p.els['rt-gtranslate-open'], 'must not mount under ' + host);
});

// 6. /chat.html still maps to / (COEP document never proxied).
var chatHref = run({ store: { 'rathor-lang': 'fr' }, path: '/chat.html' }).win.rtGTranslateHref('fr');
assert(chatHref === 'https://translate.google.com/translate?sl=en&tl=fr&u=' + encodeURIComponent('https://rathor.ai/'), 'chat must map to /: ' + chatHref);

console.log('W4c google no-same-lang checks passed');
