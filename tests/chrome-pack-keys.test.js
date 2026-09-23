/* Chrome packs: every locale file must carry the applied keys from i18n-chrome.js. */
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

var chrome = read('js/i18n-chrome.js');
var block = chrome.match(/var CHROME = \{([\s\S]*?)\};/);
assert(block, 'i18n-chrome.js must declare CHROME');
var KEYS = [];
block[1].replace(/(\w+)\s*:/g, function (_, key) { KEYS.push(key); });
assert(KEYS.length >= 40, 'applied chrome key list must stay complete');
assert(KEYS.indexOf('employTitle') !== -1, 'employTitle must stay applied');
assert(KEYS.indexOf('gTranslateNote') !== -1, 'gTranslateNote must stay applied');
assert(KEYS.indexOf('navEmploy') !== -1, 'navEmploy must stay applied');

var HIRING = {
  ar: 'التوظيف',
  es: 'Empleo',
  fr: 'Emploi',
  he: 'העסקה',
  hi: 'नियोजन',
  id: 'Pekerjakan',
  it: 'Impiego',
  ja: '任用',
  ko: '고용',
  pl: 'Zatrudnienie',
  pt: 'Emprego',
  th: 'การว่าจ้าง',
  tr: 'İstihdam',
  vi: 'Tuyển dụng',
  zh: '任用',
  el: 'Απασχόληση'
};

var RTL = { ar: 1, fa: 1, he: 1 };
var RTL_RE = /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/;

function loadPack(lang) {
  var sandbox = { window: { translations: {} } };
  vm.createContext(sandbox);
  vm.runInContext(read('i18n/' + lang + '.js'), sandbox, { filename: lang + '.js' });
  var pack = sandbox.window.translations[lang];
  assert(pack && typeof pack === 'object', lang + ' pack must assign window.translations.' + lang);
  return pack;
}

var files = fs.readdirSync(path.join(root, 'i18n')).filter(function (f) { return f.endsWith('.js'); });
assert(files.length === 23, 'product uses 23 locale packs, found ' + files.length);

var en = loadPack('en');
assert(en.employCta === 'Open how to employ →', 'English employ CTA must say how to employ');
assert(en.employCta.indexOf('employment guide') === -1, 'English employ CTA must not say employment guide');
assert(en.weekTitle === 'Recent Updates', 'English recent-updates title stays undated');
KEYS.forEach(function (key) {
  assert(typeof en[key] === 'string' && en[key].trim() !== '', 'en ' + key + ' must be a non-empty string');
});

files.forEach(function (file) {
  var lang = file.replace(/\.js$/, '');
  var pack = loadPack(lang);
  KEYS.forEach(function (key) {
    assert(typeof pack[key] === 'string' && pack[key].trim() !== '', lang + ' missing chrome key ' + key);
  });
  if (HIRING[lang]) {
    assert(pack.navEmploy !== HIRING[lang], lang + ' navEmploy must not stay the hiring label');
    assert(pack.employCta.indexOf(HIRING[lang]) === -1, lang + ' employCta must not keep the hiring label');
  }
  if (lang !== 'en') {
    assert(pack.employTitle !== en.employTitle, lang + ' employTitle must be translated');
    assert(pack.employSubtitle !== en.employSubtitle, lang + ' employSubtitle must be translated');
    assert(pack.gTranslateBtn !== en.gTranslateBtn, lang + ' gTranslateBtn must be translated');
    assert(pack.gTranslateNote !== en.gTranslateNote, lang + ' gTranslateNote must be translated');
  }
  if (RTL[lang]) {
    assert(RTL_RE.test(pack.employTitle), lang + ' employTitle must be RTL script');
    assert(RTL_RE.test(pack.navEmploy), lang + ' navEmploy must be RTL script');
    assert(RTL_RE.test(pack.gTranslateNote), lang + ' gTranslateNote must be RTL script');
  }
  assert(pack.weekTitle.indexOf('2026-09-08') === -1, lang + ' must not date Recent Updates');
});

var sw = read('sw.js');
files.forEach(function (file) {
  assert(sw.indexOf("'/i18n/" + file + "'") !== -1, 'sw precache must list /i18n/' + file);
});
assert(sw.indexOf("'/js/i18n-chrome.js'") !== -1, 'sw precache must list i18n-chrome.js');
assert(sw.indexOf('ignoreSearch: true') !== -1, 'offline pack loads must ignore the cache query');
assert(sw.indexOf("pathname.indexOf('/i18n/')") !== -1, 'offline fallback must cover /i18n/');
assert(sw.indexOf('20260923c') !== -1, 'service worker lock must match the pack token');
assert(read('js/site-lock-2026-08-22.js').indexOf('20260923c') !== -1, 'site-lock must load the same pack token');
assert(read('i18n/README.md').indexOf('20260923c') !== -1, 'i18n README must name the cache token');
assert(chrome.indexOf('rtApplyChromeI18n') !== -1, 'chrome helper must keep apply');
assert(read('js/site-lock-2026-08-22.js').indexOf('rtApplyChromeI18n') !== -1, 'site-lock must delegate chrome apply');

var pages = [
  'index.html', 'employ.html', 'privacy.html', 'contact.html', 'chat.html',
  'go-x.html', 'Launch-Ra-Thor.html', 'constellation-week.html', 'offline.html',
  'thanks.html', 'briefing.html', 'science-watches.html'
];
pages.forEach(function (page) {
  var html = read(page);
  assert(html.indexOf('/js/i18n-chrome.js?v=20260923c') !== -1, page + ' must load i18n-chrome at the pack token');
  assert(html.indexOf('/i18n/en.js?v=20260923c') !== -1, page + ' must load the English pack at the pack token');
});
assert(read('privacy.html').indexOf('data-i18n="navPrivacy"') !== -1, 'privacy title must be chrome');
assert(read('offline.html').indexOf('data-i18n="navHome"') !== -1, 'offline home control must be chrome');
assert(read('go-x.html').indexOf('data-i18n="xTitle"') !== -1, 'go-x title must be chrome');
assert(read('employ.html').indexOf('article class="rt-prose"') !== -1, 'employ essay must stay a prose article');
assert(read('science-watches.html').indexOf('2026-09-08') === -1, 'science watches must not keep the dated week window');

var essayJs = read('js/i18n-essay.js');
var essayBlock = essayJs.match(/var ESSAY = \{([\s\S]*?)\};/);
assert(essayBlock, 'i18n-essay.js must declare ESSAY');
var ESSAY = [];
essayBlock[1].replace(/(\w+)\s*:/g, function (_, key) { ESSAY.push(key); });
assert(ESSAY.length >= 200, 'visitor essay key list must cover the wired pages, found ' + ESSAY.length);
ESSAY.forEach(function (key) {
  assert(KEYS.indexOf(key) === -1, key + ' is chrome and must not also be listed as an essay key');
  assert(typeof en[key] === 'string' && en[key].trim() !== '', 'en essay key ' + key + ' must be a non-empty string');
});

var essayPages = ['index.html', 'employ.html', 'privacy.html', 'briefing.html', 'contact.html', 'Launch-Ra-Thor.html', 'go-x.html', 'science-watches.html'];
var wired = {};
essayPages.forEach(function (page) {
  var html = read(page);
  assert(html.indexOf('/js/i18n-essay.js?v=20260923c') !== -1, page + ' must load i18n-essay at the pack token');
  var marks = html.match(/data-i18n="([^"]+)"/g) || [];
  marks.forEach(function (raw) {
    var key = raw.slice('data-i18n="'.length, -1);
    if (ESSAY.indexOf(key) !== -1) wired[key] = page;
  });
});
ESSAY.forEach(function (key) {
  assert(wired[key], 'essay key ' + key + ' must be marked data-i18n on a visitor page');
});
assert(read('employ.html').indexOf('article class="rt-prose"') !== -1, 'employ essay must stay a prose article');
assert(essayJs.indexOf('rtApplyEssayI18n') !== -1, 'essay helper must apply keys');
assert(essayJs.indexOf("documentElement") === -1, 'essay apply must not set html dir');
assert(essayJs.indexOf("getElementById('rt-family-nav')") !== -1, 'essay apply must keep the family row');
assert(essayJs.indexOf("getElementById('lang-selector')") !== -1, 'essay apply must keep language tabs');
assert(read('sw.js').indexOf("'/js/i18n-essay.js'") !== -1, 'sw precache must list i18n-essay.js');
assert(read('i18n/README.md').indexOf('Operator documents') !== -1, 'readme must keep operator docs in English');
assert(read('i18n/README.md').indexOf('faqA8') !== -1, 'readme must keep the faqA8 lock');

function claimFaults(text) {
  var faults = [];
  if (/EW2 solved\s*=\s*True/i.test(text)) faults.push('EW2 solved = True');
  if (/METR\s*=\s*True/i.test(text)) faults.push('METR = True');
  if (/\bis METR\b/i.test(text)) faults.push('is METR');
  if (/Combined AGSi\s*=\s*(?!SURMISE\b)\S+/i.test(text)) faults.push('Combined AGSi = not SURMISE');
  if (/Combined AGSi is (?:true|real|a fact|fact)\b/i.test(text)) faults.push('Combined AGSi-as-fact');
  var re = /METR/g;
  var m;
  while ((m = re.exec(text))) {
    var window = text.slice(Math.max(0, m.index - 80), m.index + 8);
    if (!/≠|!=|not |Not |no measured|No measured/i.test(window)) faults.push('METR-as-fact:' + window.replace(/\s+/g, ' '));
  }
  return faults;
}

var faqA8En = 'No. A future resource-based economy via Powrush is a design intent. The lattice is built for long compatibility. That is direction, not a present economic fact.';
assert(en.faqA8 === faqA8En, 'English faqA8 must stay the design-intent lock');

files.forEach(function (file) {
  var lang = file.replace(/\.js$/, '');
  var pack = loadPack(lang);
  ESSAY.forEach(function (key) {
    assert(typeof pack[key] === 'string' && pack[key].trim() !== '', lang + ' missing essay key ' + key);
  });
  assert(typeof pack.faqA8 === 'string' && pack.faqA8.trim() !== '', lang + ' faqA8 must stay non-empty');
  assert(pack.faqA8.indexOf('royalties dissolve') === -1, lang + ' faqA8 must not restore the old royalties line');
  assert(pack.faqA8.indexOf('RBE already') === -1, lang + ' faqA8 must not say RBE already');
  var blob = Object.keys(pack).map(function (k) { return String(pack[k]); }).join('\n');
  var faults = claimFaults(blob);
  assert(faults.length === 0, lang + ' pack claim fault: ' + faults[0]);
});

console.log('chrome-pack-keys: ok (' + files.length + ' packs, ' + KEYS.length + ' chrome keys, ' + ESSAY.length + ' essay keys)');
