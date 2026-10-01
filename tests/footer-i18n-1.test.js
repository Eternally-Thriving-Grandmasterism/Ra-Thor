/* FOOTER-I18N-1: the shared fallback footer (js/family-nav-2026-08-22.js siteFooter())
 * follows the chosen language using EXISTING pack keys only.
 * Checks: the inserted footer elements carry the pack keys, those keys exist in all
 * 23 packs, no pack file changed (sha256 pinned at main d6b17afd), English text is
 * unchanged, and /pilot.html gets no hooks. */
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
  ['footerTrademarksText', "'<p class=\"rt-legal\"' + fk('footerTrademarksText') + '>Ra-Thor™ is a trademark of Autonomicity Games Inc.<br>Grok is a trademark of xAI. X is a trademark of X Corp.<br>Ra-Thor is independent — not affiliated with, sponsored by, or endorsed by xAI.</p>'"],
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
 '<div>© 2026 Sherif Samy Botros — sole steward of Autonomicity Games Inc. & AlphaProMega Air Foundation. TOLC 8.</div>'
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

// 4. No pack file changed (sha256 of main d6b17afd).
var PACKS = {
  'ar.js': '0c8a89eb0ff042043815a8c29d01093a132754f7f0e002d3ac5dfbc0801755b3',
  'de.js': '6601fed3020873156673a2dd30524992ab61121192ddfb731d7c13c90753b622',
  'el.js': 'e3981ac263109210c69e85c9ab816a277d4a0793b5c48186a2f9c6292467a8d5',
  'en.js': '5d1f24e475dcb0ac42633dd408577fb98cd30f2b2a6b28ee9f14bd2ff5634622',
  'es.js': '314aa483106ea1eec282d1c9753fbb2bf152719e1ecf8be9c3be4e65b3094892',
  'fa.js': 'cbcb9dc9c1f66a75df37d112ee859896bc1699f42a7bf573ee96b39535be8619',
  'fr.js': 'bd8007322ff21bd193e98f134b26b1ee1579eff1f262be934c0835b7b4c6c1d7',
  'he.js': 'f2b236decd5b041a4f3ec100573882745f051b209c861073532012ff09ebd08e',
  'hi.js': '7070c1c5d3d69047f7d364878bad0dbe595365ed620f0102dc15086eec0194e6',
  'id.js': '489a436b376efe1df8f7cf8fe2ec136ce5fe502f082d24cb91e23a49ecb7ef19',
  'it.js': 'd2db8208a9f668ed5baf4d3b8e0558e677f145fcc775044b226829633bf1961c',
  'ja.js': 'bdacb5e78b106cf742cf824573fc0ce81c6fd9947ef27407b3ec9e81a2d44e38',
  'ko.js': '228dce6a207f79bf6d7eb11c81e95a3522756c440c9beec64801910f92efcc01',
  'nl.js': '33b3b51d098e62e5e0d78e55bc6622f7fde804d698d7d8231116038a82963dd0',
  'pl.js': '9d02ff8da176bb8e608d5c6f7d61c89b66c6c94adb58018190a7ddac2ce4df9d',
  'pt.js': 'b846db14a6ae6c4319c31f52a18a1e9e6ce88f35814dc672bca949f7f9439d16',
  'ru.js': 'b2f9411760f894fbdc9d7bab242513359685d29274705da032b369f5ce5d2c6f',
  'sv.js': '9b1f282011275ece9db3aec97792102010c0f08ff22525ef973e02b8ee038a90',
  'th.js': '531232c3ff0905e275f66692200c2886067fbaacdd73d8aea248c74183338b23',
  'tr.js': '6e812005f189bdde5a966268ad26613c51a5b9c8ad02ba8a2e86f38bca074dcf',
  'uk.js': 'f2f213e02e882b1c060d749acb5d11ae4a3499e2cbc73b14dd7e70f1e74ea7a0',
  'vi.js': '32408cd4b909252052ef7efc0e98e08386b8b244ad187a76fc8f1b34a31fb818',
  'zh.js': 'ea4452db33765e55d350effb016a900ba2b7ea80ae0c51a41111be48f62aef1b',
};
assert(Object.keys(PACKS).length === 23, 'pack manifest must list 23 packs');
files.forEach(function (f) {
  var h = crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'i18n', f))).digest('hex');
  assert(PACKS[f] === h, 'pack changed: i18n/' + f);
});

console.log('footer-i18n-1: ' + EXPECT.length + ' footer hooks, keys in all ' + files.length + ' packs, packs unchanged, pilot excluded');
