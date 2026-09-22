/* Recent Updates: no dated week window, chrome keys translated in every pack. */
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

var home = read('index.html');
var detail = read('constellation-week.html');
var weekJs = read('js/week-window.js');
var weekJson = read('js/week-window.json');
var chrome = read('js/i18n-chrome.js');

assert(home.indexOf('Recent Updates') !== -1, 'homepage must title the section Recent Updates');
assert(home.indexOf('data-week-window') === -1, 'homepage must not paint a week window');
assert(home.indexOf('2026-09-08') === -1 && home.indexOf('2026-09-15') === -1, 'homepage must not stamp Sep 8–15');
assert(home.indexOf('data-i18n="weekLineWrap"') !== -1, 'homepage must mark the WRAP line');
assert(home.indexOf('EW2 solved = False') !== -1, 'homepage must keep EW2 solved = False');
assert(home.indexOf('This week in the constellation') === -1, 'homepage must drop the weekly title');
assert(detail.indexOf('Recent Updates') !== -1, 'detail page must say Recent Updates');
assert(detail.indexOf('data-week-window') === -1, 'detail page must not paint a week window');
assert(detail.indexOf('2026-09-08') === -1 && detail.indexOf('2026-09-15') === -1, 'detail page must not stamp Sep 8–15');
assert(weekJs.indexOf('2026-09-08') === -1 && weekJs.indexOf('2026-09-15') === -1, 'week-window.js must not hard-code the old window');
assert(weekJs.indexOf('fetch(') === -1, 'week-window.js must not fetch a date stamp');
assert(weekJson.indexOf('2026-09-08') === -1 && weekJson.indexOf('2026-09-15') === -1, 'week-window.json must not keep the old window');
assert(chrome.indexOf('weekLineWrap: 1') !== -1, 'chrome must apply weekLineWrap');
assert(chrome.indexOf('weekLineResearch: 1') !== -1, 'chrome must apply weekLineResearch');
assert(chrome.indexOf("key === 'weekLineResearch'") === -1, 'weekLineResearch must not stay a long-copy skip');
assert(home.indexOf('/i18n/en.js?v=20260922b') !== -1, 'homepage pack cache buster must move');
assert(home.indexOf("s.src = '/i18n/' + lang + '.js?v=20260922b'") !== -1, 'lang tabs must load the new pack token');

var KEYS = ['weekTitle', 'weekLead', 'weekLineWrap', 'weekLinePowrush', 'weekLineRa', 'weekLineResearch', 'weekMore'];
var HOME_LANGS = ['en', 'ar', 'es', 'fr', 'nl', 'de', 'zh', 'ja', 'pt', 'ru', 'hi'];
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

var en = loadPack('en');
assert(en.weekTitle === 'Recent Updates', 'English title must be Recent Updates');
assert(en.weekLineWrap.indexOf('EW2 solved = False') !== -1, 'English WRAP line must say EW2 solved = False');
assert(en.weekLineWrap.indexOf('solved = True') === -1 && en.weekLineWrap.indexOf('EW2 solved = True') === -1, 'must not claim EW2 solved');

var files = fs.readdirSync(path.join(root, 'i18n')).filter(function (f) { return f.endsWith('.js'); });
files.forEach(function (file) {
  var lang = file.replace(/\.js$/, '');
  var pack = loadPack(lang);
  KEYS.forEach(function (key) {
    assert(typeof pack[key] === 'string' && pack[key].trim() !== '', lang + ' ' + key + ' must be a non-empty string');
  });
  assert(pack.weekLineWrap.indexOf('EW2 solved = False') !== -1, lang + ' must keep EW2 solved = False');
  assert(pack.weekLineWrap.indexOf('COMMERCIAL_LICENSE') !== -1, lang + ' must name COMMERCIAL_LICENSE');
  assert(pack.weekLinePowrush.indexOf('Title Online') !== -1, lang + ' Powrush line must keep Title Online');
  assert(pack.weekTitle.indexOf('2026-09-08') === -1 && pack.weekLead.indexOf('2026-09-') === -1, lang + ' must not date the section');
  if (lang !== 'en') {
    assert(pack.weekTitle !== en.weekTitle, lang + ' weekTitle must not stay English');
    assert(pack.weekLineRa !== en.weekLineRa, lang + ' weekLineRa must not stay English');
    assert(pack.weekLinePowrush !== en.weekLinePowrush, lang + ' weekLinePowrush must not stay English');
    assert(pack.weekLineWrap !== en.weekLineWrap, lang + ' weekLineWrap must not stay English');
    assert(pack.weekMore !== en.weekMore, lang + ' weekMore must not stay English');
    assert(pack.weekLineResearch !== en.weekLineResearch, lang + ' weekLineResearch must not stay English');
  }
  if (RTL[lang]) {
    assert(RTL_RE.test(pack.weekTitle), lang + ' title must be RTL script');
    assert(RTL_RE.test(pack.weekLineWrap), lang + ' WRAP line must be RTL script');
    assert(RTL_RE.test(pack.weekLineRa), lang + ' Ra-Thor line must be RTL script');
  }
});

HOME_LANGS.forEach(function (lang) {
  assert(files.indexOf(lang + '.js') !== -1, 'homepage lang ' + lang + ' must have a pack');
});

console.log('recent-updates-i18n: ok (' + files.length + ' packs)');
