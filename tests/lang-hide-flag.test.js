/* LANG-HIDE-FLAG: single-flag revert checks for family-nav-only and dual-page scenarios.
 *
 * Covers:
 *   0. Static: master flag is in family-nav; i18n-chrome has no local RT_LANG_HIDE.
 *   1-4. flag=false on a family-nav-only page: no class, 'fr' kept, GT allowed.
 *   5. flag=false with family-nav absent (i18n-chrome only): 'fr' kept, no class.
 *   6. flag=true on a dual-page: 'fr' cleared, class present.
 */
var fs = require('fs');
var path = require('path');
var vm = require('vm');
var root = path.join(__dirname, '..');
var navSrc    = fs.readFileSync(path.join(root, 'js/family-nav-2026-08-22.js'), 'utf8');
var chromeSrc = fs.readFileSync(path.join(root, 'js/i18n-chrome.js'), 'utf8');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

/* ── 0. Static guards ───────────────────────────────────────────────────────── */

assert(
  /window\.RT_LANG_HIDE\s*=\s*true\s*;/.test(navSrc),
  'family-nav must define window.RT_LANG_HIDE = true (master flag)'
);

assert(
  chromeSrc.indexOf('var RT_LANG_HIDE') === -1,
  'i18n-chrome must NOT define a local RT_LANG_HIDE variable; master flag is in family-nav'
);

assert(
  chromeSrc.indexOf('window.RT_LANG_HIDE === true') !== -1,
  'i18n-chrome bootChrome must clear localStorage only when window.RT_LANG_HIDE === true (strict)'
);

assert(
  navSrc.indexOf('window.RT_LANG_HIDE === true') !== -1,
  'bootLangHide must clear localStorage only when window.RT_LANG_HIDE === true (strict)'
);

assert(
  chromeSrc.indexOf("window.addEventListener('rathor-nav-ready'") !== -1,
  "i18n-chrome must listen for rathor-nav-ready on window (family-nav dispatches on window)"
);

/* ── helpers ────────────────────────────────────────────────────────────────── */

function makeEl(tag) {
  var el = {
    tagName: (tag || '').toUpperCase(),
    id: '', src: '', href: '', rel: '', defer: false, type: '', onload: null,
    style: { cssText: '', overflowX: '', contain: '' },
    textContent: '', innerHTML: '',
    classList: {
      _list: [],
      add:    function (c) { if (el.classList._list.indexOf(c) === -1) el.classList._list.push(c); },
      remove: function (c) { el.classList._list = el.classList._list.filter(function (x) { return x !== c; }); },
      contains: function (c) { return el.classList._list.indexOf(c) !== -1; },
      toggle: function (c, force) { if (force === false) el.classList.remove(c); else el.classList.add(c); }
    },
    getAttribute:    function (k) { return el._attrs && k in el._attrs ? el._attrs[k] : null; },
    setAttribute:    function (k, v) { (el._attrs = el._attrs || {})[k] = v; },
    removeAttribute: function (k) { if (el._attrs) delete el._attrs[k]; },
    querySelectorAll: function () { return []; },
    querySelector:   function () { return null; },
    addEventListener: function () {},
    insertBefore: function () {},
    appendChild: function (child) {
      if (child && child.src) el._scripts = (el._scripts || []), el._scripts.push(child.src);
      if (child && child.href) el._scripts = (el._scripts || []), el._scripts.push(child.href);
    },
    firstChild: null,
    children: { length: 0 },
    parentNode: null
  };
  return el;
}

function makeWin(opts) {
  var store = Object.assign({}, opts.store || {});
  var htmlEl = makeEl('html');
  var head   = makeEl('head');
  var body   = makeEl('body');
  head.appendChild = function (child) {
    if (child && child.src) (opts.headScripts = opts.headScripts || []).push(child.src);
    if (child && child.href) (opts.headScripts = opts.headScripts || []).push(child.href);
  };
  body.appendChild = function (child) {
    if (child && child.src) (opts.bodyScripts = opts.bodyScripts || []).push(child.src);
  };
  var win = {
    document: {
      documentElement: htmlEl,
      head: head,
      body: body,
      readyState: opts.readyState || 'complete',
      getElementById:   function () { return null; },
      querySelector:    function () { return null; },
      querySelectorAll: function () { return []; },
      createElement:    function (t) { return makeEl(t); },
      addEventListener: function () {}
    },
    localStorage: {
      getItem:    function (k) { return k in store ? store[k] : null; },
      setItem:    function (k, v) { store[k] = String(v); },
      removeItem: function (k) { delete store[k]; }
    },
    navigator: { standalone: false },
    matchMedia:    function () { return { matches: false }; },
    dispatchEvent: function () {},
    location:      { pathname: opts.path || '/' },
    addEventListener: function () {}
  };
  win.window = win;
  if ('RT_LANG_HIDE' in opts) win.RT_LANG_HIDE = opts.RT_LANG_HIDE;
  return { win: win, store: store, htmlEl: htmlEl, opts: opts };
}

/* ── 1–4. flag=false on a family-nav-only page ──────────────────────────────── */

var flagOffSrc = navSrc.replace('window.RT_LANG_HIDE = true;', 'window.RT_LANG_HIDE = false;');
assert(flagOffSrc !== navSrc, 'src replacement failed: master flag line not found in expected form');

(function () {
  var ctx = makeWin({ store: { 'rathor-lang': 'fr' }, path: '/micro-moment.html' });
  vm.runInNewContext(flagOffSrc, ctx.win);

  assert(
    ctx.htmlEl.classList._list.indexOf('rt-lang-hide') === -1,
    'rt-lang-hide class must NOT be added when flag is false (family-nav-only page)'
  );

  assert(
    ctx.store['rathor-lang'] === 'fr',
    "stored 'fr' must be preserved when flag is false; got: " + ctx.store['rathor-lang']
  );

  var allScripts = (ctx.opts.headScripts || []).concat(ctx.opts.bodyScripts || []);
  assert(
    allScripts.some(function (s) { return s && s.indexOf('google-translate-optin') !== -1; }),
    'google-translate-optin.js must be allowed when flag is false'
  );
}());

/* ── 5. flag=false with family-nav absent (i18n-chrome only) ─────────────────
 * window.RT_LANG_HIDE is pre-set to false (e.g. by a future flag file).
 * bootChrome must not clear 'fr' and must not add rt-lang-hide. */
(function () {
  var ctx = makeWin({
    store: { 'rathor-lang': 'fr' },
    readyState: 'complete',   /* triggers immediate bootChrome() call */
    RT_LANG_HIDE: false
  });
  vm.runInNewContext(chromeSrc, ctx.win);

  assert(
    ctx.htmlEl.classList._list.indexOf('rt-lang-hide') === -1,
    'rt-lang-hide class must NOT be added by i18n-chrome when window.RT_LANG_HIDE is false'
  );

  assert(
    ctx.store['rathor-lang'] === 'fr',
    "i18n-chrome must not clear 'fr' when flag is false; got: " + ctx.store['rathor-lang']
  );
}());

/* ── 6. flag=true on a dual page: family-nav clears 'fr' ────────────────────
 * navSrc has window.RT_LANG_HIDE = true; bootLangHide must add the class and remove 'fr'. */
(function () {
  var ctx = makeWin({ store: { 'rathor-lang': 'fr' }, path: '/index.html' });
  vm.runInNewContext(navSrc, ctx.win);

  assert(
    ctx.htmlEl.classList._list.indexOf('rt-lang-hide') !== -1,
    'rt-lang-hide class must be added by family-nav when flag is true'
  );

  assert(
    !('rathor-lang' in ctx.store),
    "stored 'fr' must be cleared by bootLangHide when flag is true; got: " + ctx.store['rathor-lang']
  );
}());

console.log('lang-hide-flag: all cases pass (flag=false family-nav-only, flag=false chrome-only, flag=true dual-page)');
