// NAV-LINEAGE-1: the shared top nav (family-nav) lists Lineage and Examples,
// every public page loads family-nav, and en has a navLineage key.
// NAV-I18N-CHROME-1: every other pack translates navExamples and navLineage,
// the top bar carries those data-i18n keys, and a non-chrome page applies them.
'use strict';
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');
const root = path.join(__dirname, '..');
const nav = fs.readFileSync(path.join(root, 'js/family-nav-2026-08-22.js'), 'utf8');
const links = nav.slice(nav.indexOf('var LINKS = ['), nav.indexOf('];', nav.indexOf('var LINKS = [')));
assert.ok(/href: '\/lineage\.html'/.test(links), 'top nav LINKS must include /lineage.html');
assert.ok(/href: '\/examples\.html'/.test(links), 'top nav LINKS must include /examples.html');
for (const gone of ['Launch-Ra-Thor', 'sovereign-shard', 'web-forge']) {
  assert.ok(links.indexOf('/' + gone + '.html') === -1, gone + ' lives under Examples, not the top bar');
  assert.ok(nav.indexOf('<a href="/' + gone + '.html"') !== -1, gone + ' stays in the fallback footer');
  assert.ok(fs.readFileSync(path.join(root, 'examples.html'), 'utf8').indexOf('href="/' + gone + '.html"') !== -1, 'examples.html links ' + gone);
}
assert.ok(nav.indexOf("'<a href=\"/lineage.html\"' + fk('navLineage') + '>Lineage</a>'") !== -1, 'fallback footer hooks navLineage');
assert.ok(nav.indexOf("'<a href=\"/examples.html\"' + fk('navExamples') + '>Examples</a>'") !== -1, 'fallback footer hooks navExamples');
assert.ok(nav.indexOf("'/examples.html': 'navExamples'") !== -1, 'top bar maps examples to navExamples');
assert.ok(nav.indexOf("'/lineage.html': 'navLineage'") !== -1, 'top bar maps lineage to navLineage');
assert.ok(nav.indexOf("a.setAttribute('data-i18n', navKey)") !== -1, 'top bar links carry data-i18n');
const PUBLIC = ['index.html', 'chat.html', 'employ.html', 'pilot.html', 'Launch-Ra-Thor.html',
  'micro-moment.html', 'sovereign-shard.html', 'web-forge.html', 'contact.html', 'privacy.html',
  'licence.html', 'lineage.html', 'examples.html', 'science-watches.html', 'constellation-week.html', 'briefing.html', 'go-x.html'];
for (const f of PUBLIC) {
  const html = fs.readFileSync(path.join(root, f), 'utf8');
  assert.ok(/\/js\/family-nav-2026-08-22\.js/.test(html), f + ' must load family-nav (top nav with Lineage)');
}
const en = fs.readFileSync(path.join(root, 'i18n/en.js'), 'utf8');
assert.ok(/"navLineage": "Lineage"/.test(en), 'en navLineage key');
assert.ok(/"navExamples": "Examples"/.test(en), 'en navExamples key');

const chrome = fs.readFileSync(path.join(root, 'js/i18n-chrome.js'), 'utf8');
assert.ok(/navExamples:\s*1/.test(chrome), 'chrome allowlist includes navExamples');
assert.ok(/navLineage:\s*1/.test(chrome), 'chrome allowlist includes navLineage');
assert.ok(chrome.indexOf("'/examples.html': 'navExamples'") !== -1, 'chrome stamps examples');
assert.ok(chrome.indexOf("'/lineage.html': 'navLineage'") !== -1, 'chrome stamps lineage');

const RTL = { ar: 1, fa: 1, he: 1 };
const RTL_RE = /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/;
function loadPack(lang) {
  const sandbox = { window: { translations: {} } };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(root, 'i18n/' + lang + '.js'), 'utf8'), sandbox, { filename: lang + '.js' });
  return sandbox.window.translations[lang];
}
const packs = fs.readdirSync(path.join(root, 'i18n')).filter((f) => /^[a-z]{2}\.js$/.test(f)).sort();
assert.strictEqual(packs.length, 23, 'expected 23 locale packs');
const enPack = loadPack('en');
assert.strictEqual(enPack.navExamples, 'Examples');
assert.strictEqual(enPack.navLineage, 'Lineage');
for (const file of packs) {
  const lang = file.slice(0, 2);
  const pack = loadPack(lang);
  assert.strictEqual(typeof pack.navExamples, 'string', lang + ' navExamples');
  assert.strictEqual(typeof pack.navLineage, 'string', lang + ' navLineage');
  assert.ok(pack.navExamples.trim() !== '' && pack.navLineage.trim() !== '', lang + ' labels must be non-empty');
  if (lang !== 'en') {
    assert.notStrictEqual(pack.navExamples, enPack.navExamples, lang + ' navExamples must not stay English');
    assert.notStrictEqual(pack.navLineage, enPack.navLineage, lang + ' navLineage must not stay English');
  }
  if (RTL[lang]) {
    assert.ok(RTL_RE.test(pack.navExamples), lang + ' navExamples must be RTL script');
    assert.ok(RTL_RE.test(pack.navLineage), lang + ' navLineage must be RTL script');
  }
}

const siteLabels = {
  en: ['Examples', 'Lineage'],
  ar: ['الأمثلة', 'النسب'],
  es: ['Ejemplos', 'Linaje'],
  fr: ['Exemples', 'Lignée'],
  nl: ['Voorbeelden', 'Afstamming'],
  de: ['Beispiele', 'Abstammung'],
  zh: ['示例', '谱系'],
  ja: ['実例', '系譜'],
  pt: ['Exemplos', 'Linhagem'],
  ru: ['Примеры', 'Родословная'],
  hi: ['उदाहरण', 'वंश']
};
const site = fs.readFileSync(path.join(root, 'ra-thor-website-1.html'), 'utf8');
for (const lang of Object.keys(siteLabels)) {
  const m = site.match(new RegExp('^\\s+' + lang + ': (\\{.*\\}),?\\s*$', 'm'));
  assert.ok(m, 'ra-thor-website-1.html dictionary has ' + lang);
  const obj = JSON.parse(m[1]);
  assert.strictEqual(obj.navExamples, siteLabels[lang][0], lang + ' website navExamples');
  assert.strictEqual(obj.navLineage, siteLabels[lang][1], lang + ' website navLineage');
}

function makeEl(tag) {
  return {
    tagName: tag,
    id: '',
    href: '',
    textContent: '',
    innerHTML: '',
    className: '',
    style: {},
    children: [],
    _attrs: {},
    classList: {
      add() {},
      remove() {},
      contains() { return false; },
      toggle() {}
    },
    setAttribute(k, v) {
      this._attrs[k] = String(v);
      if (k === 'id') this.id = String(v);
    },
    getAttribute(k) {
      if (k in this._attrs) return this._attrs[k];
      if (k === 'href') return this.href || null;
      return null;
    },
    removeAttribute(k) { delete this._attrs[k]; },
    appendChild(child) { this.children.push(child); return child; },
    insertBefore(child) { this.children.unshift(child); return child; },
    querySelector() { return null; },
    querySelectorAll(sel) {
      if (sel === 'a[data-i18n]') return this.children.filter((c) => c._attrs && c._attrs['data-i18n']);
      if (sel === 'a[href]') return this.children.filter((c) => c.tagName === 'a' || c.href);
      return [];
    },
    addEventListener() {}
  };
}
(function applyOnFamilyNavOnlyPage() {
  const fr = loadPack('fr');
  const store = { 'rathor-lang': 'fr' };
  const ids = {};
  const body = makeEl('body');
  const htmlEl = makeEl('html');
  const head = makeEl('head');
  function remember(node) {
    if (node && node.id) ids[node.id] = node;
    return node;
  }
  const origAppend = body.appendChild;
  body.appendChild = function (child) { origAppend.call(this, child); return remember(child); };
  const origInsert = body.insertBefore;
  body.insertBefore = function (child) { origInsert.call(this, child); return remember(child); };
  head.appendChild = function (child) { return child; };
  const win = {
    RT_LANG_HIDE: false,
    document: {
      documentElement: htmlEl,
      head: head,
      body: body,
      readyState: 'complete',
      getElementById(id) { return ids[id] || null; },
      querySelector() { return null; },
      querySelectorAll() { return []; },
      createElement(tag) { return makeEl(tag); },
      addEventListener() {}
    },
    localStorage: {
      getItem(k) { return Object.prototype.hasOwnProperty.call(store, k) ? store[k] : null; },
      setItem(k, v) { store[k] = String(v); },
      removeItem(k) { delete store[k]; }
    },
    navigator: { standalone: false },
    matchMedia() { return { matches: false }; },
    dispatchEvent() {},
    location: { pathname: '/micro-moment.html' },
    addEventListener() {},
    translations: { en: enPack, fr: fr }
  };
  win.window = win;
  const src = nav.replace('window.RT_LANG_HIDE = true;', 'window.RT_LANG_HIDE = false;');
  vm.runInNewContext(src, win);
  const bar = ids['rt-family-nav'];
  assert.ok(bar, 'family nav mounted on a page without i18n-chrome');
  const byKey = {};
  bar.querySelectorAll('a[data-i18n]').forEach((a) => { byKey[a.getAttribute('data-i18n')] = a; });
  assert.strictEqual(byKey.navExamples && byKey.navExamples.textContent, 'Exemples');
  assert.strictEqual(byKey.navLineage && byKey.navLineage.textContent, 'Lignée');
  assert.strictEqual(byKey.navExamples.getAttribute('data-i18n'), 'navExamples');
  assert.strictEqual(byKey.navLineage.getAttribute('data-i18n'), 'navLineage');
  const ar = loadPack('ar');
  win.translations.ar = ar;
  win.rtApplyFooterI18n('ar');
  assert.strictEqual(byKey.navExamples.textContent, ar.navExamples);
  assert.strictEqual(byKey.navLineage.textContent, ar.navLineage);
  assert.strictEqual(byKey.navExamples.getAttribute('dir'), 'rtl');
  assert.strictEqual(byKey.navLineage.getAttribute('dir'), 'rtl');
  assert.strictEqual(bar.getAttribute('dir'), 'ltr');
  win.rtApplyFooterI18n('en');
  assert.strictEqual(byKey.navExamples.textContent, 'Examples');
  assert.strictEqual(byKey.navLineage.textContent, 'Lineage');
})();

console.log('nav-lineage-1: ok');
