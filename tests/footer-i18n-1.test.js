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
  'ar.js': '87113b6d50e5cfc53195822e76bb5b894d382dc5f9f16ba3c041e734d5f81bac',
  'de.js': 'f2e4151ffa5bbc7f05db92d4dd019a4b0141ec8665b3450c4c351609af8e1fb0',
  'el.js': 'ed3998642830b76c6a4bdd12e45f0af595fd8e8e370f561f82851e91cd1b5083',
  'en.js': 'a94113d019146395e6c455ca76ac3eada5c603febbb0967c9a3b348096c42140',
  'es.js': '6c232bb0f85b3bff53cd5c1124fe69aa99bb1eec7792aee1a40cfe86b270e360',
  'fa.js': 'a49885ab2764bbc1f12ec8aa5d83fa25f2f599aa8828312a7c4d67b7014a25d5',
  'fr.js': '799719545bd8b2e3dfeaa882bbc7ba203dcdc1bfc19f13823cc32d86534f6548',
  'he.js': '64a5c33afdfd4566800b2568ce479c59c108d5e69a6b0d524b9459fd8fa7d9bb',
  'hi.js': '32b607fd22da4106225718c3d9147bfd6fad1ffb9c4951738d9b3b3914189613',
  'id.js': 'c6db105270e49aede1da6eb50c2148182e0c78c09049d270d5e17a74139c8201',
  'it.js': 'e486240cbd1eed03a7c167518666b2e6514aed267dbff08c8ec56cf32b10f8c2',
  'ja.js': '74de4cd6b133b57fbe8ad24c154c0241e289c7fdefb4c220322fa74df529eb0b',
  'ko.js': 'c34ceee13b9e3bd179c84aac41ac8d8748ded2139dd70fcc70c52fb203365559',
  'nl.js': '0e07ae56044867dc1afd1a240f16fff3cf485b903f78a7ca4e9e699b29fc8d57',
  'pl.js': 'bf94979507ce7c8eb0fc2b2c3e06ddf3549cab5fa3e685ff84a02c4238712d45',
  'pt.js': 'a27ca289ecefec79d409b5ef392122b06e3df5ad44aad6bfe191d6b18acda5d2',
  'ru.js': 'c0d414cc7e8d6bbfd8250261881892488995207327934fdb3b68aa1cd1d99bba',
  'sv.js': '514242808bb29c40590eb7907a337a9345249c3a59954a71eda4b9b0b8ed95a0',
  'th.js': '66363597061e665d237a70552db417951c62ba386a51d710ee101fa364b70f1c',
  'tr.js': 'e3c713850fc92a90800aa8595574ace3a133390e72feb06477691b9970fd9353',
  'uk.js': '3749c09112361a6bf2513c369a95aa1832679c19fb57ffcd3e5aa43539f2cacc',
  'vi.js': '585c38a8857eab3920a2c8816ea9ae2e7495fa2a97d21bd538d435cfcb52ed0b',
  'zh.js': '91c91344bf6ee503258d0f176015e81aa817d2fe17d28731d01ec74ce5314744',
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
