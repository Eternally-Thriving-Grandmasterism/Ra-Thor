/* FOOTER-I18N-1: the shared fallback footer (js/family-nav-2026-08-22.js siteFooter())
 * follows the chosen language using EXISTING pack keys only.
 * Checks: the inserted footer elements carry the pack keys, those keys exist in all
 * 23 packs, pack bytes stay pinned (sha256 moved with the mercy-check wording), English footer
 * hooks stay unchanged, and /pilot.html gets no hooks. */
var fs = require('fs');
var path = require('path');
var vm = require('vm');
var crypto = require('crypto');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}
function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

var src = read('js/family-nav-2026-08-22.js');
var start = src.indexOf('  function siteFooter() {');
var end = src.indexOf('\n  }\n', start);
assert(start !== -1 && end !== -1, 'siteFooter() not found');
var footer = src.slice(start, end);

// 1. Every translatable footer element carries its pack key, next to its original English.
var EXPECT = [
  ['footerTrademarksTitle', "'<h4' + fk('footerTrademarksTitle') + '>Trademarks</h4>'"],
  ['footerTrademarksText', "'<p class=\"rt-legal\"' + fk('footerTrademarksText') + '>Ra-Thor™ is a trademark of Autonomicity Games Inc.<br>Grok is a trademark of xAI. X is a trademark of X Corp.</p>'"],
  ['footerPrivacyTitle', "'<h4' + fk('footerPrivacyTitle') + '>Privacy</h4>'"],
  ['homeFooterPrivacy', "'<p class=\"rt-legal\"' + fk('homeFooterPrivacy') + '>This website collects no personal data. Computations stay in your browser. No cookies, tracking, or analytics we control.</p>'"],
  ['footerWorkspaceTitle', "'<h4' + fk('footerWorkspaceTitle') + '>Workspace</h4>'"],
  ['homeFooterWorkspace', "'<p class=\"rt-legal\"' + fk('homeFooterWorkspace') + '>v14.15.6 · AG-SML v1.1 · TOLC 8 · Capable · Bounded · Corrigible</p>'"],
  ['footerFamilyTitle', "'<h4' + fk('footerFamilyTitle') + '>Directory</h4>'"],
  ['navHome', "'<a href=\"/\"' + fk('navHome') + '>Home</a>'"],
  ['employTitle', "'<a href=\"/employ.html\"' + fk('employTitle') + '>How to employ</a>'"],
  ['navPilot', "'<a href=\"/pilot.html\"' + fk('navPilot') + '>Pilot</a>'"],
  ['homeLaunchMap', "'<a href=\"/Launch-Ra-Thor.html\"' + fk('homeLaunchMap') + '>Launch map</a>'"],
  ['homeMoments', "'<a href=\"/micro-moment.html\"' + fk('homeMoments') + '>Micro-moments</a>'"],
  ['surfaceShard', "'<a href=\"/sovereign-shard.html\"' + fk('surfaceShard') + '>Sovereign Shard</a>'"],
  ['navContact', "'<a href=\"/contact.html\"' + fk('navContact') + '>Contact</a>'"],
  ['navPrivacy', "'<a href=\"/privacy.html\"' + fk('navPrivacy') + '>Privacy</a>'"],
  ['homeMonorepo', "rel=\"noopener\"' + fk('homeMonorepo') + '>Monorepo</a>'"]
];
EXPECT.forEach(function (pair) {
  assert(footer.indexOf(pair[1]) !== -1, 'footer element for ' + pair[0] + ' is missing its hook or its English changed');
});
var hooked = (footer.match(/fk\('([A-Za-z0-9]+)'\)/g) || []).map(function (s) { return s.slice(4, -2); });
assert(hooked.length === EXPECT.length, 'unexpected hook count: ' + hooked.length);
// Lines without a matching pack key stay English and unhooked.
['<a href="/chat.html">Lattice Chat</a>', '<a href="/web-forge.html">Web-Forge</a>',
 '<div>© 2026 Sherif Samy Botros — sole steward of Autonomicity Games Inc. & AlphaProMega Air Foundation. TOLC 8.'
].forEach(function (s) { assert(footer.indexOf(s) !== -1, 'unkeyed line changed: ' + s); });
// No data-i18n in the fallback footer: the chrome/essay/site-lock appliers must never touch it.
assert(footer.indexOf('data-i18n') === -1, 'siteFooter() must not use data-i18n');

// 2. The applier is wired and pilot is excluded.
assert(src.indexOf("var FOOTER_I18N = here !== '/pilot.html';") !== -1, 'pilot guard missing');
assert(/return FOOTER_I18N \? ' data-rt-footer-i18n="' \+ key \+ '"' : '';/.test(src), 'fk() must emit nothing on pilot');
assert(src.indexOf("document.addEventListener('rt-chrome-i18n'") !== -1, 'footer does not follow rt-chrome-i18n');
assert(/ensureFollowStrip\(\);\n\s+bootFooterI18n\(\);/.test(src), 'bootFooterI18n() not called from mount()');
assert(read('pilot.html').indexOf('data-rt-footer-i18n') === -1, 'pilot.html must not carry footer hooks');

// 3. Every key exists, non-empty, in all 23 packs.
var files = fs.readdirSync(path.join(root, 'i18n')).filter(function (f) { return /^[a-z]{2}\.js$/.test(f); }).sort();
assert(files.length === 23, 'expected 23 packs, found ' + files.length);
var sb = { window: { translations: {} } };
vm.createContext(sb);
files.forEach(function (f) { vm.runInContext(read('i18n/' + f), sb, { filename: f }); });
var T = sb.window.translations;
files.forEach(function (f) {
  var lang = f.slice(0, 2);
  assert(T[lang], lang + ': pack did not register');
  EXPECT.concat([['followTitle'], ['swatchTitle']]).forEach(function (pair) {
    var v = T[lang][pair[0]];
    assert(typeof v === 'string' && v.trim() !== '', lang + ': missing key ' + pair[0]);
  });
});

// 4. Fingerprints are pinned to the i18n pack text after LLMS-NO-GATE-WORDING-1 (#702, main e6e85a37).
var PACKS = {
  'ar.js': '397a2709e53c29a3c2384f0860b10399d84da0e5416414efab89cd72cfdf77ac',
  'de.js': '4b031b45b44ad65a1e5eb7db043587da0f0bc2265c589e50f352ab8e1895f7fb',
  'el.js': 'b0c61e6878c260c187fb5317b2f8808caf86e3708e9031730bdeba628b907975',
  'en.js': '9c742edca25c5987d9812d778bbc7848a2d3b6fdffaf383e802fc6e89b30b2dd',
  'es.js': '027661e6fb095c7213f578953d8e761b732d5f34f1f55748424338c6d5572283',
  'fa.js': '0a8f11ff1c7c0c5fc270c8989ce177b03c9ebea09db20dad8f5bf1d96f5b549e',
  'fr.js': 'd894d94d7de828c10a8a451bedeabb768235dcf9609ca30578d36da81bab063e',
  'he.js': 'ffee1e5ca7f952ec92fe0215007e3b045632c520bbebafbb299d27e96415d8bf',
  'hi.js': 'df9cf6e5d3cbf472147de9b6984388afcdc5c16afe367daa5c9ed4bfe20f816d',
  'id.js': 'f3556feba7b3423e094a4ff31aef84e40958d6e9319bc83ee661b94fd5e8bb6b',
  'it.js': '6d934bc13d69b245c39287a417277bbd60732a8825cb4304b39a0d7981190af9',
  'ja.js': '905e0a95a84f4b554db0c3ce537d4bab47444a644cf07ee3a27882ec97297983',
  'ko.js': '6ca18dd0f2624d894f4ace4fa65b1907ce991a47a8254f53a1628ac524a50d89',
  'nl.js': 'b2660d6b3c97ca6c8aab7689faa0d7ca8447803e9ca2610a75ee816cef6ff0d4',
  'pl.js': '02ee764dffb3af3a4b7533c87a778cdfc9ef832161484fb35b33bd69e409e4b6',
  'pt.js': 'f9cde5160c3e60b61cf4b808a1c735843223514cd8b4d0ac63b4e130de288bd6',
  'ru.js': '2d6bbc5b23b477b2a790ed356e44ac3d6cd4097649bac47df9d5b6ecf4ea795a',
  'sv.js': '60d9ee773af7c1141339b230b423ff8ac21fce1a9f3ff44b22f87b56b1d87039',
  'th.js': '01e66d235edb084aea02999469f4ad596617176319af0c8bcde2e4eaa05e9573',
  'tr.js': '0a3ac820b79861c29623b9480b8103655874fabad5e41b35e5f5cff482a76f89',
  'uk.js': 'abedfe2d84be428ca9e73608d9b28bee8fe62702edcbf5b12cd3a2bdb7ebbe08',
  'vi.js': 'c0daf778d69c87271ecd8c4c88c5a6a1bdb4b119d5b1d75c887c8962b460969b',
  'zh.js': 'f2728b090fad9757205e4b28dd3bb7bc3cbcac6d50bab95ae7b368ad851038bd',
};
assert(Object.keys(PACKS).length === 23, 'pack manifest must list 23 packs');
files.forEach(function (f) {
  var h = crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'i18n', f))).digest('hex');
  assert(PACKS[f] === h, 'pack changed: i18n/' + f);
});

// 5. XAI-LINE-1: the footer and every pack keep the trademark credit and carry no xAI-independence disclaimer.
assert(footer.indexOf('Grok is a trademark of xAI. X is a trademark of X Corp.') !== -1, 'footer must keep the trademark credit');
assert(!/independent|affiliated|endorsed/i.test(footer), 'footer must not carry an xAI-independence disclaimer');
files.forEach(function (f) {
  var p = T[f.slice(0, 2)];
  assert((p.footerTrademarksText.match(/<br>/g) || []).length === 1, f + ': footerTrademarksText must be the two trademark lines only');
  assert(!('faqQ18' in p) && !('faqA18' in p), f + ': the xAI-affiliation FAQ must be gone');
});

console.log('footer-i18n-1: ' + EXPECT.length + ' footer hooks, keys in all ' + files.length + ' packs, packs pinned, pilot excluded');
