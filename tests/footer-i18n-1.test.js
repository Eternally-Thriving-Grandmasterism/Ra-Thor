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
  ['footerTrademarksText', "'<p class=\"rt-legal\"' + fk('footerTrademarksText') + '>AlphaProMega is a trademark of AlphaProMega Media Inc. Powrush, Autonomicity Games and the Autonomicity Games logo are trademarks of Autonomicity Games Inc. Ra-Thor™ is a trademark of Autonomicity Games Inc. (application pending). <br>Grok is a trademark of xAI. X is a trademark of X Corp.</p>'"],
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
  ['homeMonorepo', "rel=\"noopener\"' + fk('homeMonorepo') + '>Source on GitHub</a>'"]
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
// WP-FAQ-LINK-1: faqA22 links /docs/WHITEPAPER_v4.2.html (label v4.2) in every pack.
// LINKS-FIX-1: English homeMonorepo is "Source on GitHub". Other packs keep their existing translations.
// HOME-ORDER-1: every pack gains standaloneTitle (the home heading that replaced "Start here").
// WHITEPAPER-V43-1: every pack's current-paper mentions move v4.2 → v4.3 (faqA22 link, homePaperLink, weekLine*, homeSurfacePaper);
// English faqA22 label drops "(living cover)"; English homeSurfacePaperNote names v4.3 as current (other packs untouched). No new keys.
var PACKS = {
  'ar.js': 'e54ec776b4e9acec2c6aac766ee8c29adacd39f88339ecf370b2927cf6018a6d',
  'de.js': '712e997c2755d98065e5b1f0d574e6d1872ed7065256c1afd476569621a6172b',
  'el.js': '3c676400a7d93d3c18de3fedf96167317c3a31297daf74ded754e6f502460cf2',
  'en.js': '46b3b60edd19647f43bd206daf796594dfbcc25b2b09a4e1cda65f8e899297d6',
  'es.js': '572ce90a569409b7bba0884a46646dd0cd1edac258953895660e5fafdeefae18',
  'fa.js': 'b3f6b36d8875e95dcf57ac1554f922f0c273c80e9cc69c992242d58e9d5cc5a2',
  'fr.js': '65c52040661c886712ea185d703b60ad0eb371d4c3a0ec49deedcf3097c79387',
  'he.js': '5fabe917ddb30a7e4b9efc88a61ef4018a36cac10e503e5823e66583bb5d4115',
  'hi.js': '786537fed9e7e6ecc8a8773f94289d7a8d509116bcf0d89398b116427cc38348',
  'id.js': '26245b6ab8aede2a1fffc00235b8432cdc8d4c12bab0db75a99d6cf81b865243',
  'it.js': '9122783e7092056f1d311cdc3e5eb175296e2e5248fee3fc2e9cfe1a955658e6',
  'ja.js': '8ae2aa3b6ced80ccdfe4b24425c49cb458ce62a81d932a878b2b06bd04a50774',
  'ko.js': 'eff9424a8a7567d2181a807b185f05d6fd68ce91079fb0ccb2da5dcce3631c9a',
  'nl.js': '6592f3434ab5965d36dc67f731e7960a2769280603991d3e3712a961cf509fd7',
  'pl.js': 'c0999f65aa03b2889aa1e739e823163f7c32ac24d11217e1e7dd1bf626b45b2e',
  'pt.js': '99f402024400e97b551f49a97c3ddc0cf393c15c9cc98a65fbcdc5b74d923515',
  'ru.js': 'da77949ccfb69fb3b480b01d9c65cc024fe7a4172c7272eee432c5c070f8060a',
  'sv.js': '6903d19dae89d91f658f809ac8516447698a02a72f8cbffe990b48b0ed43a806',
  'th.js': 'f59d212ef8849d20d8a98c45bf23ead0ed1a7fcd924ae3897d1d46d95954b1c8',
  'tr.js': 'a042934b82899b5e22c4141a73e7d84484ed342818c0fb099d2b0f29d04e48e0',
  'uk.js': '85f67694f4b452d74a6fee488dd3ff3f4cd49c232ffd252c54ac5247452c4ba4',
  'vi.js': '76092eb03c476afc411e10a7e8e13e1ef6fbfeb7d45ef0d8db8a34983bf3121a',
  'zh.js': '31789cd102c812c4f91eedc74659754f5e707dce752c74fdfe35ce9df4e98b8e',
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
