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
assert(KEYS.indexOf('steward') !== -1, 'steward must be applied chrome');
assert(KEYS.indexOf('langStoredNote') !== -1, 'lang storage note must be applied chrome');
assert(KEYS.indexOf('installCta') !== -1, 'installCta must stay applied chrome');

var PATH_KEYS = [
  'pathPlayTitle', 'pathPlayBody',
  'pathLicenseTitle', 'pathLicenseBody',
  'pathInspectTitle', 'pathInspectBody',
  'pathOfflineTitle', 'pathOfflineBody',
  'pathWrapTitle', 'pathWrapBody'
];
PATH_KEYS.forEach(function (key) {
  assert(KEYS.indexOf(key) !== -1, key + ' must be in the chrome allowlist');
});
['surfacesTitle', 'surfaceChat', 'surfaceChatNote', 'surfaceMap', 'surfaceMapNote', 'surfaceShard', 'surfaceShardNote', 'surfaceRepo', 'surfaceRepoNote', 'homeSurfacePaper', 'homeSurfacePaperNote'].forEach(function (key) {
  assert(KEYS.indexOf(key) !== -1, key + ' must be in the chrome allowlist');
});

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
assert(en.pathPlayTitle === 'Play', 'English play card title');
assert(en.pathPlayBody.indexOf('Title Online grey') !== -1, 'English play card keeps Title Online grey');
assert(en.pathLicenseBody.indexOf('info@Rathor.ai') !== -1, 'English license card keeps the contact');
assert(en.pathWrapTitle === 'WRAP-EW2', 'WRAP card title stays the repo name');
assert(en.pathWrapBody.indexOf('EW2 solved = False') !== -1, 'English WRAP card keeps EW2 solved = False');
assert(en.pathWrapBody.indexOf('COMMERCIAL_LICENSE') !== -1, 'English WRAP card keeps the commercial license');
assert(en.pathWrapBody.indexOf('No AG-SML certification') !== -1, 'English WRAP card keeps no AG-SML certification');
assert(en.langStoredNote.indexOf('rathor.ai') !== -1, 'English language note names this device and the site');
assert(en.homeSurfacePaper === 'Whitepaper v4.2', 'on-device paper card stays whitepaper v4.2');
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
    assert(pack.pathPlayTitle !== en.pathPlayTitle, lang + ' pathPlayTitle must be translated');
    assert(pack.pathPlayBody !== en.pathPlayBody, lang + ' pathPlayBody must be translated');
    assert(pack.pathLicenseTitle !== en.pathLicenseTitle, lang + ' pathLicenseTitle must be translated');
    assert(pack.pathInspectTitle !== en.pathInspectTitle, lang + ' pathInspectTitle must be translated');
    assert(pack.pathOfflineTitle !== en.pathOfflineTitle, lang + ' pathOfflineTitle must be translated');
    assert(pack.pathWrapBody !== en.pathWrapBody, lang + ' pathWrapBody must be translated');
    assert(pack.steward !== en.steward, lang + ' steward must be translated');
    assert(pack.langStoredNote !== en.langStoredNote, lang + ' langStoredNote must be translated');
    assert(pack.homeSurfacePaperNote !== en.homeSurfacePaperNote, lang + ' homeSurfacePaperNote must be translated');
  }
  assert(pack.pathWrapTitle === 'WRAP-EW2', lang + ' WRAP title stays the repo name');
  assert(pack.pathPlayBody.indexOf('Title Online') !== -1, lang + ' play card must keep Title Online');
  assert(pack.pathWrapBody.indexOf('EW2 solved = False') !== -1, lang + ' WRAP card must keep EW2 solved = False');
  assert(pack.pathWrapBody.indexOf('COMMERCIAL_LICENSE') !== -1, lang + ' WRAP card must keep COMMERCIAL_LICENSE');
  assert(pack.pathWrapBody.indexOf('solved = True') === -1, lang + ' WRAP card must not solve EW2');
  if (RTL[lang]) {
    assert(RTL_RE.test(pack.employTitle), lang + ' employTitle must be RTL script');
    assert(RTL_RE.test(pack.navEmploy), lang + ' navEmploy must be RTL script');
    assert(RTL_RE.test(pack.gTranslateNote), lang + ' gTranslateNote must be RTL script');
    assert(RTL_RE.test(pack.pathPlayTitle), lang + ' pathPlayTitle must be RTL script');
    assert(RTL_RE.test(pack.pathWrapBody), lang + ' pathWrapBody must be RTL script');
    assert(RTL_RE.test(pack.steward), lang + ' steward must be RTL script');
    assert(RTL_RE.test(pack.langStoredNote), lang + ' langStoredNote must be RTL script');
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
assert(sw.indexOf('20260923f') !== -1, 'service worker lock must match the pack token');
assert(read('js/site-lock-2026-08-22.js').indexOf('20260923f') !== -1, 'site-lock must load the same pack token');
assert(read('i18n/README.md').indexOf('20260923f') !== -1, 'i18n README must name the cache token');
assert(chrome.indexOf('rtApplyChromeI18n') !== -1, 'chrome helper must keep apply');
assert(read('js/site-lock-2026-08-22.js').indexOf('rtApplyChromeI18n') !== -1, 'site-lock must delegate chrome apply');

var pages = [
  'index.html', 'employ.html', 'privacy.html', 'contact.html', 'chat.html',
  'go-x.html', 'Launch-Ra-Thor.html', 'constellation-week.html', 'offline.html',
  'thanks.html', 'briefing.html', 'science-watches.html'
];
pages.forEach(function (page) {
  var html = read(page);
  assert(html.indexOf('/js/i18n-chrome.js?v=20260923f') !== -1, page + ' must load i18n-chrome at the pack token');
  assert(html.indexOf('/i18n/en.js?v=20260923f') !== -1, page + ' must load the English pack at the pack token');
});
var homeHtml = read('index.html');
PATH_KEYS.forEach(function (key) {
  assert(homeHtml.indexOf('data-i18n="' + key + '"') !== -1, 'homepage must mark ' + key);
});
assert(homeHtml.indexOf('data-i18n="homeLaunchMap"') !== -1, 'homepage quick link must mark Launch map');
assert(homeHtml.indexOf('data-i18n="homeMoments"') !== -1, 'homepage quick link must mark Micro-moments');
assert(homeHtml.indexOf('data-i18n="homeOfflineChat"') !== -1, 'homepage quick link must mark Offline Lattice Chat');
assert(homeHtml.indexOf('data-i18n="installCta"') !== -1, 'homepage install control must mark installCta');
assert(homeHtml.indexOf('data-i18n="steward"') !== -1, 'homepage steward line must stay marked');
assert(read('js/pwa-install.js').indexOf('installCta') !== -1, 'install button path must read installCta');
assert(read('js/rathor-feedback.js').indexOf('langStoredNote') !== -1, 'language note must read langStoredNote');
assert(read('js/site-lock-2026-08-22.js').indexOf('Operator bench') === -1, 'site-lock must not restamp English product paths');
assert(read('js/site-lock-2026-08-22.js').indexOf('pathPlayTitle') !== -1, 'site-lock must keep the play-card key');
assert(read('js/site-lock-2026-08-22.js').indexOf('homeSurfacePaper') !== -1, 'on-device paper card must be marked');
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
assert(ESSAY.indexOf('footerTrademarksTitle') !== -1, 'footer trademarks title must be an essay key');
assert(ESSAY.indexOf('footerTrademarksText') !== -1, 'footer trademarks text must be an essay key');
ESSAY.forEach(function (key) {
  assert(KEYS.indexOf(key) === -1, key + ' is chrome and must not also be listed as an essay key');
  assert(typeof en[key] === 'string' && en[key].trim() !== '', 'en essay key ' + key + ' must be a non-empty string');
});

var essayPages = ['index.html', 'employ.html', 'privacy.html', 'briefing.html', 'contact.html', 'Launch-Ra-Thor.html', 'go-x.html', 'science-watches.html'];
var wired = {};
essayPages.forEach(function (page) {
  var html = read(page);
  assert(html.indexOf('/js/i18n-essay.js?v=20260923f') !== -1, page + ' must load i18n-essay at the pack token');
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

var CHAT_KEYS = [
  'chatTitle', 'chatSubtitle', 'chatOfflineMercy', 'chatPathFast', 'chatPathServer',
  'chatLocalIntel', 'chatStatusDefault', 'chatNotAvailable', 'chatLocalNote',
  'chatSearch', 'chatSpeak', 'chatSend', 'chatSessionFoot',
  'chatBridgeTitle', 'chatBridgeBody', 'chatCopyTitle', 'chatCopyContext',
  'chatOpenGrok', 'chatOpenX',
  'chatReplyHello', 'chatReplyWho', 'chatReplyTolc', 'chatReplyPrivacy',
  'chatReplyOffline', 'chatReplyLocal', 'chatReplyOllama', 'chatReplyDoc',
  'chatReplySearch', 'chatReplyLicense', 'chatReplyPowrush', 'chatReplyCopy',
  'chatReplyHelp', 'chatReplyThanks', 'chatReplyBye', 'chatReplyMercy',
  'chatReplyEmpty', 'chatReplyFallback'
];
CHAT_KEYS.forEach(function (key) {
  assert(KEYS.indexOf(key) !== -1, key + ' must be in the chrome allowlist');
  assert(ESSAY.indexOf(key) === -1, key + ' must not be an essay key');
});
assert(en.chatTitle === 'Lattice Chat ⚡️', 'English chat title stays Lattice Chat');
assert(en.chatSubtitle.indexOf('14.15.6') !== -1, 'English chat subtitle keeps workspace 14.15.6');
assert(en.chatOfflineMercy === 'Offline Mercy Thunder', 'English offline mercy label');
assert(en.chatPathFast === 'Fast Responder', 'English path badge stays Fast Responder');
assert(en.chatPathServer === 'Local Server', 'English local server label');
assert(en.chatLocalIntel === 'Local Intelligence', 'English local intelligence label');
assert(en.chatNotAvailable === 'Not available', 'English WebLLM unavailable label');
assert(en.chatSpeak === 'Speak your truth…', 'English composer placeholder');
assert(en.chatSend === 'Send', 'English send label');
assert(en.chatSearch === 'Search…', 'English search placeholder');
assert(en.chatCopyContext === 'Copy Context for any LLM', 'English copy-context label');
assert(en.chatCopyTitle === 'Copy Context', 'English copy-context title');
assert(en.chatOpenGrok === 'Open Grok Demo', 'English Open Grok label');
assert(en.chatOpenX === 'Open X Demo', 'English Open X label');
assert(en.chatBridgeTitle === 'Bridge to any cloud LLM', 'English bridge title');
assert(en.chatSessionFoot.indexOf('No backend we control') !== -1, 'English session footer');
assert(en.chatReplyHello.indexOf('Thunder locked in, Mate') !== -1, 'English hello keeps Thunder locked in, Mate');
assert(en.chatReplyPowrush.indexOf('inspect ≠ METR') !== -1, 'canned Powrush line keeps inspect ≠ METR');
assert(en.chatReplyPowrush.indexOf('EW2 solved = True') === -1, 'canned replies must not solve EW2');
assert(en.chatStatusDefault === 'Fast responder active (default)', 'English default status');

var chatHtml = read('chat.html');
var chatJs = read('js/chat.js');
CHAT_KEYS.forEach(function (key) {
  if (key.indexOf('chatReply') === 0 || key === 'chatNotAvailable') return;
  assert(chatHtml.indexOf('data-i18n="' + key + '"') !== -1, 'chat.html must mark ' + key);
});
assert(chatHtml.indexOf('data-i18n-attr="placeholder"') !== -1, 'chat placeholders must use data-i18n-attr');
assert(chatJs.indexOf('chatStr(') !== -1, 'chat.js must read pack strings');
assert(chatJs.indexOf('Thunder locked in') === -1, 'canned hello must come from the pack, not a chat.js literal');
assert(chatJs.indexOf("'chatReplyHello'") !== -1, 'hello canned line must use chatReplyHello');
assert(chatJs.indexOf("'chatReplyFallback'") !== -1, 'fallback canned line must use chatReplyFallback');
assert(chatJs.indexOf("'chatNotAvailable'") !== -1, 'Not available must be read from the pack');
assert(chatJs.indexOf('Reply in ') !== -1, 'preamble may include Reply in {language}');
assert(chatJs.indexOf('systemPreamble()') !== -1, 'copy context and model calls must use the preamble');
assert(chatJs.indexOf('generateLocalResponse') !== -1, 'fast responder must stay');
var responder = chatJs.slice(chatJs.indexOf('function generateLocalResponse'), chatJs.indexOf('function setBackendUI'));
assert(responder.indexOf('replyInClause') === -1, 'Fast Responder must not become a translator');
assert(responder.indexOf('Reply in ') === -1, 'canned responder must not append Reply in');
assert(chatJs.indexOf("getElementById('rt-family-nav')") !== -1, 'chat dir apply must keep the family row');
assert(chatJs.indexOf("getElementById('lang-selector')") !== -1, 'chat dir apply must keep language tabs');
assert(chatJs.indexOf('chat-messages') !== -1 && chatJs.indexOf('chatInput') !== -1, 'transcript and input take surface dir');
assert(chatJs.indexOf('ceo@acitygames.com') === -1, 'chat.js must not print ceo@acitygames.com');
assert(chatHtml.indexOf('ceo@acitygames.com') === -1, 'chat.html must not print ceo@acitygames.com');
assert(read('i18n/README.md').indexOf('does not speak 23 languages') !== -1, 'readme must not claim a 23-language responder');

var chromeSandbox = {
  document: {
    readyState: 'loading',
    addEventListener: function () {},
    documentElement: { setAttribute: function () {} },
    body: null,
    getElementById: function () { return null; },
    querySelector: function () { return null; },
    querySelectorAll: function () { return []; }
  }
};
chromeSandbox.window = chromeSandbox;
vm.createContext(chromeSandbox);
vm.runInContext(chrome, chromeSandbox, { filename: 'i18n-chrome.js' });
assert(typeof chromeSandbox.rtChatSurfaceDir === 'function', 'chrome must export chat surface dir');
var enDir = chromeSandbox.rtChatSurfaceDir(en.chatSpeak, 'ar');
assert(enDir.dir === 'ltr' && enDir.lang === 'en', 'English chat copy stays ltr even when rathor-lang is ar');
var arDir = chromeSandbox.rtChatSurfaceDir('مرحبا', 'ar');
assert(arDir.dir === 'rtl' && arDir.lang === 'ar', 'Arabic script on the chat surface is rtl');
var faDir = chromeSandbox.rtChatSurfaceDir('سلام', 'fa');
assert(faDir.dir === 'rtl' && faDir.lang === 'fa', 'Persian script on the chat surface is rtl');
var heDir = chromeSandbox.rtChatSurfaceDir('שלום', 'he');
assert(heDir.dir === 'rtl' && heDir.lang === 'he', 'Hebrew script on the chat surface is rtl');

files.forEach(function (file) {
  var lang = file.replace(/\.js$/, '');
  if (lang === 'en') return;
  var pack = loadPack(lang);
  CHAT_KEYS.forEach(function (key) {
    assert(pack[key] !== en[key], lang + ' ' + key + ' must be translated');
    if (RTL[lang]) {
      assert(RTL_RE.test(pack[key]), lang + ' ' + key + ' must be RTL script');
    }
  });
  assert(pack.chatReplyPowrush.indexOf('inspect ≠ METR') !== -1, lang + ' canned Powrush line keeps inspect ≠ METR');
  assert(pack.chatReplyPowrush.indexOf('EW2 solved = True') === -1, lang + ' canned replies must not solve EW2');
  assert(pack.chatSubtitle.indexOf('14.15.6') !== -1, lang + ' chat subtitle keeps workspace 14.15.6');
});

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
