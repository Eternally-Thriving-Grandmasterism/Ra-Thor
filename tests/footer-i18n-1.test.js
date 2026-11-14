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
  ['navExamples', "'<a href=\"/examples.html\"' + fk('navExamples') + '>Examples</a>'"],
  ['navLicence', "'<a href=\"/licence.html\"' + fk('navLicence') + '>Ra-Thor Commercial Licence</a>'"],
  ['navLineage', "'<a href=\"/lineage.html\"' + fk('navLineage') + '>Lineage</a>'"],
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
// MODEL-NOTICES-1: licenceFaqQuote and faqA16 carry the same English §1B parenthetical in every pack.
var PACKS = {
  'ar.js': '011a89f97159b11bfd469e7200bf6d3d5cb971814867422f0a934466d6195f14',
  'de.js': '6c53e0fe5584ecf5af35972a05bef43c9db1d699561a88a4078733a45e104606',
  'el.js': '8841dca7af22f4fb04b072a89bd240d49bfa834209a362cc75a20f6ae09582db',
  'en.js': 'f806b5e8421d0c0f57caab6f455954a68ac6405a694bab399d6774731f245d6b',
  'es.js': 'f4b0020f792b8d0fed17f82e0d571231b70abde64adfb4ba96d49fbf5fc03b4e',
  'fa.js': '68a03ffc11620dcf1dd96a12e14032d4be7f2b42fd387d6a7862f8b7180ab08e',
  'fr.js': '4f2a81e36c25f0f739ef432daa344e550cd1960e24339655e9c7b4fad9b9baa2',
  'he.js': 'f5a055d155e033f8067585e3ab38d0f97eabc298f27f9ba3ada73a4373c2adc1',
  'hi.js': '8a91c66d123f0b0d9af804c6ef4bbc3a240ac915c7f374e1b0121c999679d4d8',
  'id.js': '7946d944dc89e82c7606bf092f2b3fe79815175b252018b74fba1700a0ce1844',
  'it.js': '74fad5dfcced4d0a9761fa2499b004c0b315e94f7daa714e4c39b5e7c6b56522',
  'ja.js': 'f016c99696ffff06071d96ae0393c47b5727dd4f5070e4cbe01b320406c84c8d',
  'ko.js': '3d3c105aee4d0921e52b41f63623790b9c8e0595e2c3afc17ba809a9542043d0',
  'nl.js': '6e0a9a4a2e7f3b21c9a2be3842a779cf3f5799935c51f2ed3b4a3ed09e812e4e',
  'pl.js': 'ec16dddd55e50832f0cc5ee393f8eafdeb877c2e3391f1a5ba79ee6e942bb7e0',
  'pt.js': '05a4c2acf5970b9896511cbe807d9f80667b2f11545007647e930fac97f7f6fb',
  'ru.js': '789e3ea7da03a5041732cef09cb3ac2139d4892106879575a148580bc416440e',
  'sv.js': '6716834d7fc2ac9f328faf09eb4ac697f06edb9c1abd414f2c2f1c51e47216a9',
  'th.js': '54cdfa2ae1e87b900d2d0ebd557af594b9ddd7be918c3dc4c74dd5d113014029',
  'tr.js': 'c1960be1b651ea58634be3443138bf6332b0d0780965a5e0415f187e09156249',
  'uk.js': '178366e72a07f2b0aa07007e2630d0bb9bec78bfa941bf6cdf2007cc0bd5b516',
  'vi.js': '71feb55f188622aeded35e3455f9431cb3030ec676a97e6052529a9074d4a3bc',
  'zh.js': '26c7cd6b87bb9492c0c725b67e2022e4669912fd38f0c01f6b0988c7a46ed479',
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
