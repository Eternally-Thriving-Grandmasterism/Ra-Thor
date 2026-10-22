/* LANG-HIDE-FLAG: flag=false on a family-nav-only page (no i18n-chrome.js).
 * Verifies: no rt-lang-hide class on <html>, stored 'fr' is kept, Google Translate is allowed. */
var fs = require('fs');
var path = require('path');
var vm = require('vm');
var root = path.join(__dirname, '..');
var navSrc = fs.readFileSync(path.join(root, 'js/family-nav-2026-08-22.js'), 'utf8');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

// 0. Static: the master flag must live in family-nav.
assert(
  /window\.RT_LANG_HIDE\s*=\s*true\s*;/.test(navSrc),
  'family-nav must define window.RT_LANG_HIDE = true (master flag)'
);

// 1. Simulate flag=false on a family-nav-only page (replace master flag for this test).
var flagOffSrc = navSrc.replace('window.RT_LANG_HIDE = true;', 'window.RT_LANG_HIDE = false;');
assert(
  flagOffSrc !== navSrc,
  'replacement must have changed the source (master flag line not found in expected form)'
);

var htmlClasses = [];
var scriptsLoaded = [];
var store = { 'rathor-lang': 'fr' };

function makeEl(tag) {
  var el = {
    tagName: (tag || '').toUpperCase(),
    id: '', src: '', href: '', rel: '', defer: false, type: '', onload: null,
    style: { cssText: '', overflowX: '', contain: '' },
    textContent: '', innerHTML: '',
    classList: {
      add:    function (c) { htmlClasses.push(c); },
      remove: function (c) { htmlClasses = htmlClasses.filter(function (x) { return x !== c; }); },
      contains: function (c) { return htmlClasses.indexOf(c) !== -1; },
      toggle: function (c, force) {
        if (force === false) el.classList.remove(c);
        else el.classList.add(c);
      }
    },
    getAttribute:    function (k) { return el[k] !== undefined ? el[k] : null; },
    setAttribute:    function (k, v) { el[k] = v; },
    removeAttribute: function (k) { delete el[k]; },
    querySelectorAll: function () { return []; },
    querySelector:   function () { return null; },
    addEventListener: function () {},
    insertBefore: function () {},
    appendChild: function (child) {
      if (child && child.src) scriptsLoaded.push(child.src);
      if (child && child.href) scriptsLoaded.push(child.href);
    },
    firstChild: null,
    children: { length: 0 },
    parentNode: null
  };
  return el;
}

var head = makeEl('head');
var body = makeEl('body');
var htmlEl = makeEl('html');

var win = {
  document: {
    documentElement: htmlEl,
    head: head,
    body: body,
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
  location:      { pathname: '/micro-moment.html' },
  addEventListener: function () {}
};
win.window = win;

vm.runInNewContext(flagOffSrc, win);

// 2. No rt-lang-hide class on <html>.
assert(
  htmlClasses.indexOf('rt-lang-hide') === -1,
  'rt-lang-hide class must NOT be added to <html> when window.RT_LANG_HIDE is false'
);

// 3. Stored 'fr' preference must be kept (not cleared).
assert(
  store['rathor-lang'] === 'fr',
  "stored 'fr' must be preserved when flag is false; got: " + store['rathor-lang']
);

// 4. Google Translate script must be allowed (loaded into head).
assert(
  scriptsLoaded.some(function (s) { return s && s.indexOf('google-translate-optin') !== -1; }),
  'google-translate-optin.js must be allowed (appended to head) when flag is false'
);

console.log('lang-hide-flag: flag=false on family-nav-only page: no rt-lang-hide class, fr kept, Google Translate allowed');
