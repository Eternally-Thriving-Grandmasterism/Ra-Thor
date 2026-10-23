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
  ['footerTrademarksText', "'<p class=\"rt-legal\"' + fk('footerTrademarksText') + '>AlphaProMega is a trademark of AlphaProMega Media Inc. Powrush, Autonomicity Games and the Autonomicity Games logo are trademarks of Autonomicity Games Inc. The Ra-Thor trademark is pending.<br>Grok is a trademark of xAI. X is a trademark of X Corp.</p>'"],
  ['footerPrivacyTitle', "'<h4' + fk('footerPrivacyTitle') + '>Privacy</h4>'"],
  ['homeFooterPrivacy', "'<p class=\"rt-legal\"' + fk('homeFooterPrivacy') + '>This website collects no personal data. Computations stay in your browser. No cookies, tracking, or analytics we control.</p>'"],
  ['footerWorkspaceTitle', "'<h4' + fk('footerWorkspaceTitle') + '>Workspace</h4>'"],
  ['homeFooterWorkspace', "'<p class=\"rt-legal\"' + fk('homeFooterWorkspace') + '>v14.15.6 · AG-SML v1.2 · TOLC 8 · Capable · Bounded · Corrigible</p>'"],
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
// Hashes updated by PRIVACY-DATE-1: privacyUpdated → 2026-10-22, employStamp → 22 Oct 2026, swatchStamp date stripped.
var PACKS = {
  'ar.js': '5e4cea59ade5a85ccb738c96c6b53dc587e1598094e91fe7779e3ae3f6efd02e',
  'de.js': 'feffe50eebedf9323f1a28925d8e6f80a953087d59b529754ba0613ae8165821',
  'el.js': '99a3c57fd9b91fe1a161c9d437cf44b9740409c85f00d3b7e827905a18f82fc7',
  'en.js': 'f24613d88644f021d196d4b6a20a9c24979e132565e145aedebd30c0b8bb5216',
  'es.js': 'dd107d70c9b77f976aa7be38456a9ea896c79b90404bcce616c1773ab4219aa6',
  'fa.js': '0fd948865bf980ed25838fc7bba943f9355cad243f2581b39168e8afcae7b5be',
  'fr.js': '5243d2cf06ef02f0730b555cda078e7c0835702ed0728888fb96cadcded88a13',
  'he.js': '457850989e0514f7c15d2b179fed60eb599f58c77dc5d1d6f958943c28832db1',
  'hi.js': '47dfa241e447191f47f18b9cedf5bc7423a25b027ceea0bdc7c8a688511c333f',
  'id.js': '7f8fd60b540fd1beedbd4e095586eb68e4588ceae670773c4f7dc40e97dc6272',
  'it.js': 'b5d5271f44383a0a0c836c4ebc05f8f246515649419cc22901a5d72f3e0cf099',
  'ja.js': '9f2122e4aa8d08e1c07ca2c167e50716a30ca186e11ff181f3f9a4f6b9aa2908',
  'ko.js': 'bf2c8c99af3b4d5e592db0f36c83620b482ac2a92537c8ba83a339a97acdc367',
  'nl.js': 'dace4d8045b779e60bd1cd5674beed6b2080b508e23fbc7a7a7201f2c40fb0a4',
  'pl.js': '8a543b233af950c2fbc0b7287c8ffcbd6ce3c9a66dfd81aefb14c60afcf33a0a',
  'pt.js': '2f88b9c7f0fdd3a362f9553a7397a7c28be2a97ac343529ea3900d1dd10b0818',
  'ru.js': 'a6c921a8f7f899ce47466d9037598a25c0923a42d56d476fceb944a32fe35277',
  'sv.js': 'ea6a7d2e140b07f063326315fcad6dfb9909e02cbc4fa5d946cfc618f728a2b8',
  'th.js': 'c4667b961ac44f0ff224d07f2935d80bb818344e6d26322c84f293d657414114',
  'tr.js': 'c98295d59bdf284f71c8872b90c1deeb170347d1e36eb34427088228d5cc6343',
  'uk.js': '15f411150c26b4e734d7e8a1c1df1655354a2282f760ca9021d3f9d4885e98f0',
  'vi.js': '1e5a09c3afbde3ae9f68ea6496ca747b502d49ac58b5f02a6a5affdb91d58443',
  'zh.js': '37f1831c3e2481e67c453731ef11a961beaa451476a9fd1952fde63356ca6918',
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
